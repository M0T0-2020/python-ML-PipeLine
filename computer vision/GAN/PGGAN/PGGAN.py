import os, sys
import pandas as pd
import numpy as np
import gc, pickle
import random, math, time
import torch
from torch import nn
from models import GNet, DNet

class PGGAN:
    def __init__(self, latent_dim, G_depthScale0, D_depthScale0, channels, min_dim=8, alpha=0.2, lr=0.0002, b1=0.5, b2=0.999):
        self.latent_dim, self.channels, self.alpha = latent_dim, channels, alpha
        self.lr, self.b1, self.b2 = lr, b1, b2

        self.G_depth_list = []
        self.D_depth_list = []
        self.G_depth_list.append(G_depthScale0)
        self.D_depth_list.append(D_depthScale0)
        
        self.generator = GNet(latent_dim, G_depthScale0, initBiasToZero=True,
                              leakyReluLeak=0.2, normalization=True, generationActivation=None, 
                              dimOutput=channels, equalizedlR=True)
        self.discriminator = DNet(D_depthScale0, initBiasToZero=True, leakyReluLeak=0.2, 
                                  sizeDecisionLayer=1, miniBatchNormalization=False, generationActivation=None,
                                  dimInput=channels, equalizedlR=True)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        self.generator.setNewAlpha(alpha)
        self.discriminator.setNewAlpha(alpha)

        self.criterion = torch.nn.BCELoss()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_cnt = 0

        self.min_dim = min_dim

    def addNewLayer(self, layer_type):
        G_nowlayerdim = self.G_depth_list[-1]
        D_nowlayerdim = self.D_depth_list[-1]
        G_newlayerdim = max(self.min_dim, G_nowlayerdim//2)
        D_newlayerdim = max(self.min_dim, D_nowlayerdim//2)

        self.generator.addScale(G_newlayerdim, layer_type)
        self.discriminator.addScale(D_newlayerdim, layer_type)
    
    def save_models(self, name=None, path=""):
        g_layer_log = '_'.join(list(map(str,self.G_depth_list)))
        d_layer_log = '_'.join(list(map(str,self.D_depth_list)))
        if not name:
            torch.save(self.generator.state_dict(), f"{path}generator{self.train_cnt}_{g_layer_log}.pt")
            torch.save(self.discriminator.state_dict(), f"{path}discriminator{self.train_cnt}_{d_layer_log}.pt")
        else:
            torch.save(self.generator.state_dict(), f"{path}generator{name}_{g_layer_log}.pt")
            torch.save(self.discriminator.state_dict(), f"{path}discriminator{name}_{d_layer_log}.pt")

    def compute_loss(self, img):
        batch_size = img.size(0)
        
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        valid = torch.Tensor(batch_size, 1).fill_(1.0).to(self.device)
        fake = torch.Tensor(batch_size, 1).fill_(0.0).to(self.device)

        
        """
        train Generator 
        """
        self.optimizer_G.zero_grad()
        z = np.random.normal(0, 1, (batch_size, self.latent_dim)).tolist()
        z = torch.FloatTensor(z).to(self.device)
        gen_imgs = self.generator(z)
        g_loss = self.criterion(self.discriminator(gen_imgs), valid)
        g_loss.backward()
        self.optimizer_G.step()
        self.generator = self.generator.to('cpu')

        """
        train Discriminator
        """
        self.optimizer_D.zero_grad()
        real_loss = self.criterion(self.discriminator(img.to(self.device)), valid)
        fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        self.discriminator = self.discriminator.to('cpu')

        del img, gen_imgs, valid, fake
        gc.collect()
        self.train_cnt+=1

        return g_loss.item(), d_loss.item()