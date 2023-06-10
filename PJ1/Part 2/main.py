import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import Mydata
import CNN


EPOCH = 500
LR = 0.001

DataSet = Mydata.MyDataset(1,520)
DataSet.load_data()
testData = Mydata.MyDataset(521,620)
testData.load_data()
testData_2 = Mydata.MyDataset(1,620)
testData_2.load_data()
train_loader = Data.DataLoader(DataSet,batch_size= 64, shuffle= True)
test_loader = Data.DataLoader(testData,batch_size = 100,shuffle= True)
test_loader_2 = Data.DataLoader(testData_2,batch_size = 1,shuffle= True)



cnn = CNN.CNN()
CNN.train(cnn,train_loader,test_loader,EPOCH,LR)

CNN.test(cnn,test_loader_2)

torch.save(cnn,"Model_4")
 