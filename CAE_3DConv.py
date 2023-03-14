import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
import numpy as np
import h5py
from argparse import ArgumentParser
import cv2
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob
from sklearn.metrics import roc_curve, auc, precision_recall_curve

WINDOWSIZE = 75
IMGSIZE = 64

class WindowDataset(Dataset):
    def __init__(self, data_file_path, label_file_path, mode):
        self.data_path = data_file_path
        self.data_len = len(glob(self.data_path+'/*'))
        self.windowSize = WINDOWSIZE
        self.mode = mode
        if self.mode=='test':
            #Reading test labels
            ff = h5py.File(label_file_path,'r')
            self.labels = np.array(ff['5_sec_labels'])
            ff.close()

    def __len__(self):
        return self.data_len

    def __getitem__(self,idx):
        imgtmp = cv2.resize(cv2.imread(self.data_path+'/'+str(idx)+'/'+str(0)+'.png', cv2.IMREAD_GRAYSCALE),(IMGSIZE,IMGSIZE))
        imgtmp = torch.from_numpy(imgtmp/255).float()
        window = imgtmp.unsqueeze(0)
        for i in range(1,self.windowSize):
            imgtmp = cv2.resize(cv2.imread(self.data_path+'/'+str(idx)+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE),(IMGSIZE,IMGSIZE))
            imgtmp = torch.from_numpy(imgtmp/255).float()
            window = torch.cat((window,imgtmp.unsqueeze(0)),0)

        if self.mode=='train':
            return window.unsqueeze(0)
        if self.mode=='test':
            return window.unsqueeze(0), self.labels[idx,:]

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Autoencoders_lightning(pl.LightningModule):
    def __init__(self,**dict_args):
        super(Autoencoders_lightning,self).__init__()
        self.save_hyperparameters()
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()

        ##encoder layers
        self.conv_block_1 = BasicConv(1,128,3, stride=1, padding=1, bn = True, relu=True)
        self.conv_block_2 = BasicConv(128,64,3, stride=1, padding=1, bn = True, relu = True)
        self.conv_block_3 = BasicConv(64,32,3, stride=1, padding=1, bn=True, relu=True)
        self.max_pool_1 = nn.MaxPool3d((3,2,2))
        self.max_pool_2 = nn.MaxPool3d((1,2,2))

        ##decoder layers
        self.dec_convtrans_1 = nn.ConvTranspose3d(32,64,3,stride=1, padding=1)
        self.dec_convtrans_2 = nn.ConvTranspose3d(64,128,3,stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.dec_convtrans_3 = nn.ConvTranspose3d(128,1,3,stride=(3,2,2), padding=(0,1,1), output_padding=(0,1,1))
        self.deconv_batch_norm_1 = nn.BatchNorm3d(64)
        self.deconv_batch_norm_2 = nn.BatchNorm3d(128)

    def forward(self,x):
        x = x.contiguous()
        code = self.encoderDecoder_3lev(x,'encoder')
        output = self.encoderDecoder_3lev(code,'decoder')
        return output

    def encoderDecoder_3lev(self,x,req):
        if req == 'encoder':
            code = self.conv_block_1(x)
            code = self.max_pool_1(code)
            code = self.conv_block_2(code)
            code = self.max_pool_2(code)
            code = self.conv_block_3(code)
            return code
        elif req == 'decoder':
            out = self.dec_convtrans_1(x)
            out = self.deconv_batch_norm_1(out)
            out = self.dec_convtrans_2(out)
            out = self.deconv_batch_norm_2(out)
            out = self.dec_convtrans_3(out)
            out = self.sig(out)
            return out

    def train_dataloader(self):
        dataset = WindowDataset(self.hparams.train_file_path, '', 'train')
        print('Path = ', dataset.data_path, ', Data Length = ', dataset.data_len)
        dataloader = DataLoader(dataset,batch_size=self.hparams.train_batch_size, num_workers = self.hparams.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset_test = WindowDataset(self.hparams.test_file_path, self.hparams.label_file_path, 'test')
        self.len_test_set = len(dataset_test)
        dataloader_test = DataLoader(dataset_test,batch_size=1,num_workers = self.hparams.num_workers)
        print('Path = ', dataset_test.data_path, ', Data Length = ', dataset_test.data_len)

        self.testRes = h5py.File('testResults.hdf5','w')
        self.loss = self.testRes.create_dataset('loss',shape=(self.len_test_set,1), maxshape=(None,1),compression='gzip',compression_opts=9,chunks=(10,1))
        self.hlabel = self.testRes.create_dataset('labels',shape=(self.len_test_set,1), maxshape=(None,1),compression='gzip',compression_opts=9,chunks=(10,1))
        return dataloader_test

    def loss_function(self, out, batch_data):
        return F.mse_loss(out, batch_data)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return self.optimizer

    def training_step(self,train_batch, batch_idx):
        z = train_batch
        #self.adjust_learning_rate(self.optimizer,self.current_epoch)
        output = self.forward(z)
        loss = self.loss_function(output, z)
        self.log('batch_train_loss', loss.detach().cpu())
        return {"loss": loss.cpu()}

    def training_epoch_end(self, outs):
        loss_list = list(x['loss'] for x in outs)
        loss = torch.stack(loss_list).mean()
        self.log_dict({"epochCalc/train_loss": loss, "step": self.current_epoch})

    def test_step(self,test_batch,batch_idx):
        z, z_label = test_batch
        output = self.forward(z)
        loss = self.loss_function(output,z)
        self.loss[batch_idx,:]=loss.cpu()
        self.hlabel[batch_idx,:]=z_label.cpu()
        self.logger.experiment.add_scalars('test', {'score': loss}, global_step=batch_idx)
        return np.concatenate((loss.reshape((1,1)).cpu().numpy(), z_label.cpu()), 1)

    def test_epoch_end(self, outs):
        vals = np.concatenate(outs, 0)
        scores, labels = vals[:,0], vals[:,1]
        fpr, tpr, _ = roc_curve(labels, scores, pos_label = 1)
        auc_roc = round(auc(fpr,tpr),4)
        prec,rec,_ = precision_recall_curve(labels, scores, pos_label = 1)
        auc_pr = round(auc(rec, prec),4)
        print("\nAUC(ROC): ", auc_roc, " AUC(PR): ", auc_pr)

    def adjust_learning_rate(self,optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
        lr = self.hparams.lr * (0.1 ** (epoch // 20))
        print('learning rate decayed to ',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main(args):
    dict_args = vars(args)
    checkpoint_callback = ModelCheckpoint(
        filename = '{epoch}',
        # save_top_k = -1,
        # every_n_epochs=2
        )
    settng = 'train' #train, trainCtd

    if settng=='train':
        # Training from beginning
        model = Autoencoders_lightning(**dict_args)
        trainer = pl.Trainer(max_epochs = args.max_epochs, gpus = int(args.gpus), strategy="ddp", callbacks=[checkpoint_callback])
        trainer.fit(model)

    elif settng=='trainCtd':
        # Resuming training from saved checkpoint
        model = Autoencoders_lightning(**dict_args)
        trainer = pl.Trainer(max_epochs = args.max_epochs, gpus = int(args.gpus), strategy="ddp", callbacks=[checkpoint_callback],
            resume_from_checkpoint='lightning_logs/version_0/checkpoints/epoch=0.ckpt')
        trainer.fit(model)

    elif settng=='test':
        # Testing
        trainer = pl.Trainer(gpus = int(args.gpus))
        print('Calculating test scores')
        model = Autoencoders_lightning.load_from_checkpoint(checkpoint_path='lightning_logs/version_0/checkpoints/epoch=2.ckpt')
        trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # parametrize the network
    parser.add_argument('--train_batch_size', type=int, default=10, help='Batch size during training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_file_path',type=str, help='Path to folder containing training video frames.')
    parser.add_argument('--test_file_path',type=str, help='Path to folder containing test video frames.')
    parser.add_argument('--label_file_path',type=str, help='Path to HDF5 file containing test labels.')
    # add all the available options to the trainer

    args = parser.parse_args()
    main(args)
