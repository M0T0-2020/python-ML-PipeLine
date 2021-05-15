import numpy as np
import torch
from torch import nn
import gc, pickle

def weights_init_normal(m):
    """
    ConvNet.apply(weights_init_normal)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.conv_init_dim = 256
        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.conv_init_dim * self.init_size ** 2))
        self.channels = channels

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.conv_init_dim),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(self.conv_init_dim, self.conv_init_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_init_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(self.conv_init_dim, self.conv_init_dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_init_dim//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.conv_init_dim//2, self.conv_init_dim//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_init_dim//4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.conv_init_dim//4, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.conv_init_dim, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        self.channels = channels
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class DCGAN:
    def __init__(self, img_size, latent_dim, channels, lr=0.0002, b1=0.5, b2=0.999):
        self.img_size, self.latent_dim, self.channels = img_size, latent_dim, channels
        self.lr, self.b1, self.b2 = lr, b1, b2
        
        self.generator = Generator(img_size, latent_dim, channels)
        self.discriminator = Discriminator(self.img_size, self.channels)

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        self.criterion = torch.nn.BCELoss()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_cnt = 0

    def save_models(self, name=None, path=""):
        if not name:
            torch.save(self.generator.state_dict(), f"{path}generator{self.train_cnt}.pt")
            torch.save(self.discriminator.state_dict(), f"{path}discriminator{self.train_cnt}.pt")
        else:
            torch.save(self.generator.state_dict(), f"{path}generator{name}.pt")
            torch.save(self.discriminator.state_dict(), f"{path}discriminator{name}.pt")

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