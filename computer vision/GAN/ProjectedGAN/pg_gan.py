from copy import deepcopy
from time import time
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader, Dataset

from pg_modules.discriminator import ProjectedDiscriminator
from pg_modules.networks_fastgan import Generator as fastgan
from pg_modules.networks_stylegan2 import Generator as stylegan2
from pj_utils import GModelType
from torch_utils.ops import upfirdn2d


def _get_generator(
    model_type: GModelType,
    z_dim=256,
    c_dim=0,
    w_dim=0,
    img_resolution=256,
    img_channels=3,
    ngf=128,
    cond=0,
    mapping_kwargs={},
):
    if model_type == GModelType.STYLEGAN2:
        gen = stylegan2(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
        )
    else:
        gen = fastgan(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            ngf=ngf,
            cond=cond,
            mapping_kwargs=mapping_kwargs,
        )
    return gen


class ProjectedGAN:
    def __init__(
        self,
        model_type: GModelType,
        device: torch.device,
        data_set: Dataset,
        accumulate_num: int,
        epoch_num: int,
        batch_size: int,
        z_dim=256,
        c_dim=0,
        w_dim=0,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0, # 0 -> Single 1->conditinal
        mapping_kwargs={},
        gen_optim_kwarg: Dict = {"lr": 5e-5},
        dis_optim_kwarg: Dict = {"lr": 5e-5},
    ) -> None:
        self.generator = _get_generator(
            model_type=model_type,
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            ngf=ngf,
            cond=cond,
            mapping_kwargs=mapping_kwargs,
        ).to(device)

        self.generator_ema = deepcopy(self.generator)
        self.discriminator = ProjectedDiscriminator(
            backbone_kwargs={"im_res": img_resolution, "num_discs": 4, "cond": cond}
        ).to(device)

        self.g_optimizer: RAdam = RAdam(
            params=self.generator.parameters(), **gen_optim_kwarg
        ).to(device)
        self.d_optimizer: RAdam = RAdam(
            params=self.discriminator.parameters(), **dis_optim_kwarg
        ).to(device)

        self.device = device
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.dl: DataLoader = DataLoader(data_set, self.batch_size, shuffle=42)
        self.accumulate_num: int = accumulate_num
        self.num_epochs: int = epoch_num

        iters = np.arange(self.num_epochs)
        self.ema_lambda_schedule = 1.0 + 0.5 * (0.98 - 1.0) * (1 + np.cos(np.pi * iters / len(iters)))

        self.LGavg, self.LDgen, self.LDreal = [], [], []

    def run_G(self, z, c):
        img = self.generator(z.to(self.device), c.to(self.device))
        return img

    def run_D(self, img, c, blur_sigma=0):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function("blur"):
                f = (
                    torch.arange(-blur_size, blur_size + 1, device=img.device)
                    .div(blur_sigma)
                    .square()
                    .neg()
                    .exp2()
                )
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.discriminator(img.to(self.device), c.to(self.device))
        return logits

    def ema_update(self, epcoh):
        m = self.ema_lambda_schedule[epcoh]
        for param_q, param_k in zip(self.generator.parameters(), self.generator_ema.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def g_fn(self, z, c):
        blur_sigma = 0
        gen_img = self.run_G(z, c)
        gen_logits = self.run_D(gen_img, c, blur_sigma=blur_sigma)
        loss_Gmain: torch.Tensor = (-gen_logits).mean()
        loss_Gmain /= self.accumulate_num
        loss_Gmain.backward()
        return loss_Gmain.item()

    def d_fn(self, real_img, real_c, z, c):
        blur_sigma = 0
        gen_img = self.run_G(z.to(self.device), c.to(self.device), update_emas=True)
        gen_logits = self.run_D(gen_img, c.to(self.device), blur_sigma=blur_sigma)
        loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()
        loss_Dgen /= self.accumulate_num
        loss_Dgen.backward()

        real_img_tmp = real_img.detach().requires_grad_(False)
        real_logits = self.run_D(
            real_img_tmp.to(self.device), real_c.to(self.device), blur_sigma=blur_sigma
        )
        loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()
        loss_Dreal /= self.accumulate_num
        loss_Dreal.backward()

        return loss_Dgen.item(), loss_Dreal.item()

    def train_fn(self, real_img, real_c):
        z = torch.randn(self.batch_size, self.z_dim)
        c = torch.randn(self.batch_size, 10)
        loss_Gmain = self.g_fn(z, c)
        loss_Dgen, loss_Dreal = self.d_fn(real_img, real_c, z, c)
        return loss_Gmain, loss_Dgen, loss_Dreal

    def train_one_epoch(self, epoch):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        l_len = len(self.dl)
        loss_Gmain_avg, loss_Dgen_avg, loss_Dreal_avg = 0, 0, 0
        for i, real_img, real_c in enumerate(self.dl):
            loss_Gmain, loss_Dgen, loss_Dreal = self.train_fn(
                real_img=real_img, real_c=real_c
            )
            loss_Gmain_avg += loss_Gmain / l_len
            loss_Dreal_avg += loss_Dreal / l_len
            loss_Dgen_avg += loss_Dgen / l_len
            it = l_len * epoch + i
            if (it + 1) % self.accumulate_num == 0:
                self.g_optimizer.step()
                self.d_optimizer.step()
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                self.ema_update(epoch)
        return loss_Gmain_avg, loss_Dgen_avg, loss_Dreal_avg

    def train(self):
        for epoch in range(self.num_epochs):
            t = time()
            LGavg, LDgen_avg, LDreal_avg = self.train_one_epoch(epoch)
            self.LGavg.append(LGavg)
            self.LDgen.append(LDgen_avg)
            self.LDreal.append(LDreal_avg)
            _t = time() - t
            print(
                f"EPOCH {epoch+1}, Loss Generator {LGavg}, Loss Dis(gen) {LDgen_avg}, Loss Dis(real) {LDreal_avg}, TIME {_t}"
            )
