import torch.utils.data as Data
import numpy as np
import copy

class MyDataset(Data.Dataset): #继承Dataset
    
    def __init__(self,data_path): #初始化一些属性
        self.DataSet = []
        self.AnsSet = []
        self.count = 0
        self.data_path = data_path

        
    def load_data(self):
        self.count = 0
        fp = open(self.data_path,"r+",encoding="utf-8")
        words = []
        tags = []
        for line in fp:
            if line == '\n':
                self.DataSet.append(copy.deepcopy(words))
                self.AnsSet.append(copy.deepcopy(tags))
                words.clear()
                tags.clear()
                self.count += 1
                continue
            items = line.split()
            words.append(items[0])
            tags.append(items[1].rstrip())
                
 
    def __len__(self):#返回整个数据集的大小
        return self.count
 
    def __getitem__(self,index):#根据索引index返回图像及标签        
        return self.DataSet[index],self.AnsSet[index]


def get_data(filepath):
    fp = open("./Part 3/train.txt","r")
    words = []
    tags = []
    train_data = []
    for line in fp:
        if line == "\n":
            senten = copy.deepcopy((words,tags))
            train_data.append(senten)
            words.clear()
            tags.clear()
            continue
        items = line.split()
        word, tag = items[0], items[1].rstrip()
        words.append(word)
        tags.append(tag)
        
dataset = MyDataset("./Part 3/Chinese/train.txt")
dataset.load_data()
print(dataset)

train_loader = Data.DataLoader(dataset,batch_size= 10, shuffle= True)
# test_loader = Data.DataLoader(dataset,batch_size = 10,shuffle= True)
for step,data in enumerate(train_loader):
    x,y = data
    print("当前{}步数".format(step,(x,y)))