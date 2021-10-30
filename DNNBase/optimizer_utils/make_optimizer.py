from torch.optim import Adam
import transformers
from LAMB import Lamb
from RAdam import RAdam
from sam import SAM

def make_optimizer(model, optimizer_name="AdamW", sam=False):
    optimizer_grouped_parameters = model.parameters()
    kwargs = {
        'lr':1e-4,
        'weight_decay':0.01,
        #"betas":(0.9, 0.999),
        # 'eps': 1e-06
    }
    if not sam:
        if optimizer_name == "RAdam":
            optimizer = RAdam(optimizer_grouped_parameters, **kwargs)
            return optimizer
        if optimizer_name == "LAMB":
            optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
            return optimizer
        elif optimizer_name == "Adam":
            from torch.optim import Adam
            optimizer = Adam(optimizer_grouped_parameters, **kwargs)
            return optimizer
        elif optimizer_name == "AdamW":
            optimizer = transformers.AdamW(optimizer_grouped_parameters, **kwargs)
            return optimizer
        else:
            raise Exception('Unknown optimizer: {}'.format(optimizer_name))
    else:
        if optimizer_name == "RAdam":
            base_optimizer = RAdam(optimizer_grouped_parameters, **kwargs)
            optimizer = SAM(base_optimizer, rho=0.05)
            return optimizer
        if optimizer_name == "LAMB":
            base_optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
            optimizer = SAM(base_optimizer, rho=0.05)
            return optimizer
        elif optimizer_name == "Adam":
            from torch.optim import Adam
            base_optimizer = Adam(optimizer_grouped_parameters, **kwargs)
            optimizer = SAM(base_optimizer, rho=0.05, **kwargs)
            return optimizer
        elif optimizer_name == "AdamW":
            from transformers import AdamW
            base_optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
            optimizer = SAM(base_optimizer, rho=0.05, **kwargs)
            return optimizer
        else:
            raise Exception('Unknown optimizer: {}'.format(optimizer_name))