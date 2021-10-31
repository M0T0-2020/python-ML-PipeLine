from torch import cuda

class DNNConfig:
    def getattr(self, name, except_values):
        try: return getattr(self, name)
        except:  return except_values
    device =  'cuda' if cuda.is_available() else 'cpu'

    epoch_num = 128
    batch_size = 64

    #optim params
    optimizer_name = 'Adam'
    issam = True
    scheduler_name = 'CosineAnnealingLR'
    lr=1e-4
    weight_decay=0
    isswa = False
    swa_lr = 0.025
    swa_start = 80


    # scheduler params
    T_max = 128

    num_warmup_steps = 100
    num_training_steps = 100

    milestones=[30, 60, 90]
    
    mode = 'min' # 'max'
    factor = 0.5