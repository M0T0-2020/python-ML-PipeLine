import torch
import pickle
import bz2
import base64

model = torch.nn.Linear(128,128)

#圧縮
state_dict = model.state_dict()
state_dict = torch.load('model.pth')
PARAM = base64.b64encode(bz2.compress(pickle.dumps(state_dict)))

#読み込み
state_dict = pickle.loads(bz2.decompress(base64.b64decode(PARAM)))
model.load_state_dict(state_dict)