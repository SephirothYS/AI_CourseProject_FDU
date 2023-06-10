import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import os
import cv2
import numpy as np

class MyDataset(Data.Dataset): #继承Dataset
    
    def __init__(self,strat,end): #初始化一些属性
        self.DataSet = np.zeros((12 * (end - strat),1,28,28))
        self.AnsSet = np.zeros((12 * (end - strat) ))
        self.count = 0
        self.end = end
        self.strat = strat
        
        
    def load_data(self):
        x = np.full((28,28), 1 / 255)
        self.count = 0
        self.DataSet.astype(np.float32)
        self.AnsSet.astype(np.float32)
        for i in range(12):
            for j in range(self.strat,self.end):
                # filename = "train/%d/%d.bmp" % (i + 1,j + 1)
                # filename = "./Part 2/train/%d/%d.bmp" % (i + 1 , j + 1)
                filename = "G:\\test_data\\%d\\%d.bmp" % (i + 1,j + 1)
                img = cv2.imread(filename)
                imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = imGray * x 
                self.DataSet[self.count][0] = result
                self.AnsSet[self.count] = i
                self.count += 1
 
    def __len__(self):#返回整个数据集的大小
        return self.count
 
    def __getitem__(self,index):#根据索引index返回图像及标签        
        return self.DataSet[index],self.AnsSet[index]

# dataset = MyDataset(1,100)
# dataset.load_data()
# print(dataset)

# train_loader = Data.DataLoader(dataset,batch_size= 10, shuffle= True)
# test_loader = Data.DataLoader(dataset,batch_size = 10,shuffle= True)
# for step,data in enumerate(train_loader):
#     x,y = data
#     print("当前{}步数".format(step,(x,y)))