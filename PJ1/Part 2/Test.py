import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import Mydata
import CNN


# testData_2 = Mydata.MyDataset(1,620)
testData_2 = Mydata.MyDataset(0,240)
testData_2.load_data()
test_loader_2 = Data.DataLoader(testData_2,batch_size = 1,shuffle= True)

cnn = torch.load("./Part 2/Model_2")
CNN.test(cnn,test_loader_2)