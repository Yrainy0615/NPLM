import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from scripts.dataset.inversion_dataset import InversionDataset
from scripts.model.deepSDF import DeepSDF
from scripts.model.point_encoder import PCAutoEncoder
import argparse
import os
import wandb
from scripts.model.loss_functions import inversion_loss

class InversionTrainer():
    def __init__(self, cfg, decoder_shape, decoder_deform, dataloader, device):
        self.cfg = cfg['training']
        self.decoder_shape = decoder_shape
        self.decoder_deform = decoder_deform
        self.trainloader = dataloader
        self.device = device
        self.encoder = PCAutoEncoder()
        self.encoder.to(device)
        self.checkpoint_path = self.cfg['save_path']
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        
    def reduce_lr(self, epoch):
        if self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'encoder_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'optimizer_encoder_state_dict': self.optimizer.state_dict(),
                       },
                       path)
       
    def train_step(self, batch):
        self.encoder.train()
        self.optimizer.zero_grad()

        loss_dict = inversion_loss(batch, self.encoder, self.decoder_shape, self.decoder_deform,self.device)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        self.optimizer.step()


        loss_dict = {k: loss_dict[k].item() for k in loss_dict.keys()}
           
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
                loss_dict = self.train_step(batch)
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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=3, help='gpu index')
    parser.add_argument('--wandb', type=str, default='inversion_latent_mse', help='run name of wandb')

    # setting
    args = parser.parse_args()
    wandb.init(project='NPLM', name =args.wandb)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfgpath = 'NPLM/scripts/configs/inversion.yaml'
    CFG = yaml.safe_load(open(cfgpath, 'r'))
    
    # initialize dataset
    dataset = InversionDataset(root_dir=CFG['training']['root_dir'],
                                 n_samples=CFG['training']['n_samples'],
                                 n_sample_noise=CFG['training']['n_sample_noise'],
                                 sigma_near=CFG['training']['sigma_near'])
    dataloader = DataLoader(dataset, batch_size=CFG['training']['batch_size'],shuffle=False, num_workers=2)
    
    # initialize for shape decoder
    decoder_shape = DeepSDF(
            lat_dim=CFG['shape_decoder']['decoder_lat_dim'],
            hidden_dim=CFG['shape_decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
        ) 
    checkpoint_shape = torch.load(CFG['training']['checkpoint_shape'])
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    
    # initialize for deformation decoder
    decoder_deform = DeepSDF(lat_dim=512+200,
                    hidden_dim=1024,
                    geometric_init=False,
                    out_dim=3,
                    input_dim=3)
    checkpoint_deform  = torch.load(CFG['training']['checkpoint_deform'])
    decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder_deform.eval()
    decoder_deform.to(device)
    
    # initialize trainer
    trainer = InversionTrainer(CFG, decoder_shape, decoder_deform, dataloader, device)
    trainer.train(10001)