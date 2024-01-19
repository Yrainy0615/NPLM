import torch
import yaml
from torch.utils.data import Dataset, DataLoader, random_split
from scripts.dataset.inversion_dataset import InversionDataset, LeafInversion, Inversion_2d
from scripts.dataset.sdf_dataset import LeafColorDataset
from scripts.model.fields import UDFNetwork
from scripts.model.point_encoder import PCAutoEncoder, Imgencoder
from scripts.model.deepUDF import NDF
import argparse
import os
import wandb
from scripts.model.loss_functions import inversion_loss, inversion_2d, inversion_weight



class InversionTrainer():
    def __init__(self, cfg, trainloader, decoder,lat_idx_all,device):
        self.cfg = cfg['training']
        self.trainloader = trainloader
        self.decoder = decoder
     #   self.testloader = testloader
        self.lat_idx_all = lat_idx_all
        self.device = device
        self.encoder = NDF()
        self.encoder.to(device)
        self.checkpoint_path = self.cfg['save_path']
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        # load a pretrained resnet 50 as the encoder
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.encoder_2d = Imgencoder(dino, 512)
        # add a fc layer to encoder_2d (374,512)

        self.encoder_2d.to(device)
        self.optimizer_2d = torch.optim.Adam(self.encoder_2d.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
    def reduce_lr(self, epoch):
        if self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'inversion_2d_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'encoder_state_dict': self.encoder_2d.state_dict(),
                        'optimizer_encoder_state_dict': self.optimizer_2d.state_dict(),
                       },
                       path)
       
    def train_step_depth(self, batch):
        self.encoder.train()
        self.optimizer.zero_grad()

        loss_dict = inversion_loss(batch, self.encoder,self.device)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        self.optimizer_2d.step()


        loss_dict = {k: loss_dict[k].item() for k in loss_dict.keys()}
           
        return loss_dict    
    
    def train_step_2d(self,batch):
        self.encoder_2d.train()
        self.optimizer_2d.zero_grad()
        loss_dict = inversion_weight(batch, self.encoder_2d,self.device, self.lat_idx_all)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        self.optimizer_2d.step()
        return loss_dict   

    def train(self, epochs):
        loss = 0
       # start = self.load_checkpoint()
        start =0
        ckpt_interval =self.cfg['ckpt_interval']
        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.trainloader:
                loss_dict = self.train_step_2d(batch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
  
                wandb.log(loss_values)
                
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckpt_interval ==0:
                self.save_checkpoint(epoch)
              
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)
            # # add test loss
            # sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            # sum_loss_dict.update({'loss':0.0})
            # for batch in self.testloader:
            #     loss_dict = inversion_loss(batch, self.encoder,self.device)
            #     # 如何区分test的loss变量名
                
            #     for k in loss_dict:
            #         sum_loss_dict[k] += loss_dict[k]
            #         test_values = {"test_" + key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}                
            #         wandb.log(test_values)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=7, help='gpu index')
    parser.add_argument('--wandb', type=str, default='inversion_latent_mse', help='run name of wandb')

    # setting
    args = parser.parse_args()
    wandb.init(project='NPLM', name =args.wandb)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfgpath = 'NPLM/scripts/configs/inversion.yaml'
    CFG = yaml.safe_load(open(cfgpath, 'r'))
    mode = '2d'
    # initialize dataset
    if mode == '3d':
        dataset = LeafInversion(root_dir=CFG['training']['root_dir'],
                           device=device, )
    if mode == '2d':
        # dataset = LeafColorDataset(mode='train',
        #                          batch_size=CFG['training']['batch_size'],
        #                          root_dir=CFG['training']['root_dir_color'])
        dataset = Inversion_2d(root_dir='sample_result/shape_new',device=device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    trainloader = DataLoader(dataset, batch_size=CFG['training']['batch_size'],shuffle=False, num_workers=2)
    #testloader = DataLoader(val_dataset, batch_size=CFG['training']['batch_size'],shuffle=False, num_workers=2)
    # initialize for shape decodedr
    # decoder_shape = DeepSDF(
    #         lat_dim=CFG['shape_decoder']['decoder_lat_dim'],
    #         hidden_dim=CFG['shape_decoder']['decoder_hidden_dim'],
    #         geometric_init=True,
    #         out_dim=1,
    #     ) 
    decoder = UDFNetwork(d_in=CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                         d_out=1,
                         n_layers=CFG['decoder']['decoder_nlayers'],
                         udf_type='sdf')
    
    checkpoint_shape = torch.load(CFG['training']['checkpoint_shape'])
    lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
    decoder.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder.eval()
    decoder.to(device)
    
    # initialize for deformation decoder
    # decoder_deform = DeepSDF(lat_dim=512+200,
    #                 hidden_dim=1024,
    #                 geometric_init=False,
    #                 out_dim=3,
    #                 input_dim=3)
    # checkpoint_deform  = torch.load(CFG['training']['checkpoint_deform'])
    # decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    # decoder_deform.eval()
    # decoder_deform.to(device)
    

    # initialize trainer
    trainer = InversionTrainer(CFG, trainloader,decoder ,lat_idx_all,device)
    trainer.train(10001)