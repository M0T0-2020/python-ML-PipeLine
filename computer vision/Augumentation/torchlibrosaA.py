 #based on https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/augmentation.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width=None, stripes_num=None):
        """Drop stripes. 
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input
        
        if self.drop_width is None:
            return input
        if self.stripes_num is None:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input


    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(self, 
                 v_drop_width=None, v_stripes_num=None,
                 h_drop_width=None, h_stripes_num=None):
        """Spec augmetation. 
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.h_dropper = DropStripes(dim=2, drop_width=h_drop_width, 
            stripes_num=h_stripes_num)

        self.v_dropper = DropStripes(dim=3, drop_width=v_drop_width, 
            stripes_num=v_stripes_num)

    def forward(self, input):
        x = self.v_dropper(input)
        x = self.h_dropper(x)
        return x

if __name__ == "__main__":
    spec_a = SpecAugmentation(
        h_drop_width=3,
        h_stripes_num=3,
        v_drop_width=2,
        v_stripes_num=4
    )
    b, w, h, d = 10, 64, 64, 3
    x = torch.rand(b, d, h, w)
    x = spec_a(x)