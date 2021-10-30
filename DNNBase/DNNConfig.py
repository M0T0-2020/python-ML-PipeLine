from torch import cuda

class DNNConfig:
    device =  'cuda' if cuda.is_available() else 'cpu'

    epoch_num = 128
    batch_size = 64

    optimizer_name = 'Adam'
    issam = True
    scheduler_name = 'CosineAnnealingLR'

    lr=1e-4
    weight_decay=0

    # scheduler params
    T_max = 128

    num_warmup_steps = 100
    num_training_steps = 100

    milestones=[30, 60, 90]
    
    mode = 'min' # 'max'
    factor = 0.5