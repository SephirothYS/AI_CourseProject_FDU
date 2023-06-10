import ResNet18
import Mydata
import torch

Net = torch.load("./Part 3/Model1")
train_loader, test_loader = Mydata.get_data(16,True)
ResNet18.test(Net,test_loader)