import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data

# im = Image.open('data/src/lenna_square.png')
# cv.imread()
class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X=X
        self.y=y
        
    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        # argumentation
        x = self.X[index]
        x = torch.FloatTensor(x)
        
        
        if self.y!=None:
            return {
                'input': x,
                'label': torch.tensor(self.y[index], dtype=torch.float)
            }
        else:
            return {
                'input': x
            }


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    #Mutually exclusive with batch_size, shuffle, sampler, and drop_last
    #batch_size, shuffle, sampler, and drop_last　と互いに排他的
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, sampling_p, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / (label_to_count[self._get_label(dataset, idx)])**(1/sampling_p)
            for idx in self.indices]
        
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        """if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError"""
        return dataset.y[idx][0]
        
                
    def __iter__(self):
        return ( self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True) )

    def __len__(self):
        return self.num_samples

class balancedBatchDatasetSampler(torch.utils.data.sampler.BatchSampler):
    #Mutually exclusive with batch_size, shuffle, sampler, and drop_last
    #batch_size, shuffle, sampler, and drop_last　と互いに排他的
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, batch_size, indices=None,  callback_get_label=None):
        self.indices = list(range(len(dataset)))
        self.batch_size = batch_size
        
        self.loader_len = len(dataset)//batch_size
        self.surplus = len(dataset)%batch_size
        self.argsort = np.argsort(dataset.y)
        self.weights = None
        
        
        self._get_group()
        self.make_batch_plan()
        
    def _get_group(self, ):
        surplus_keys = np.random.choice(range(self.batch_size), self.surplus, replace=False)
        self.group = {}
        cnt = 0
        for i in range(self.batch_size):
            if i in surplus_keys:
                size = self.loader_len + 1
            else:
                size = self.loader_len
            self.group[i] = self.argsort[cnt:cnt+size]
            cnt += size
        for value in self.group.values():
            np.random.shuffle(value)
            
        for key,value in self.group.items():
            self.group[key]=list(value)
        max_length = max(len(v) for v in self.group.values())
        for key, value in self.group.items():
            if len(value)<max_length:
                self.group[key]=value+random.sample(value,k=1)

    def make_batch_plan(self):
        self.batch_plan = []
        if self.surplus==0:
            size = self.loader_len
        else:
            size = self.loader_len + 1                
        for _ in range(size):
            l = []
            for value in self.group.values():
                if len(value)>0:
                    l.append(value.pop(0))
            self.batch_plan.append(l)
    
    """
    def __iter__(self):
        sample_batch_plan = self.batch_plan.pop(0)
        if len(self.batch_plan)==0:
            self._get_group()
            self.make_batch()
        return ( int(idx) for idx in sample_batch_plan )
    """
    def __iter__(self):
        for batch in self.batch_plan:
            yield batch
        self._get_group()
        self.make_batch_plan()

    def __len__(self):
        return self.loader_len