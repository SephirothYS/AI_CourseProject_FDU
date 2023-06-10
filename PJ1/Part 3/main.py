import ResNet18
import Mydata
import torch

LR = 0.001
lr_decay_step = 7
MAX_EPOCH = 3
BATCH_SIZE = 16

train_loader, test_loader = Mydata.get_data(BATCH_SIZE,shuffle_flag=True)
Net = ResNet18.ResNet18(1,10)
ResNet18.train(Net.model,LR,lr_decay_step,MAX_EPOCH,train_loader,test_loader)

torch.save(Net,"Model1")