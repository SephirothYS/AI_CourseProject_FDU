import os
import gzip
import torch.utils.data as Data
import numpy as np
import torch
import torchvision.transforms as transforms

class DealDataset(Data.Dataset):

    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = self.load_data(folder, data_name, label_name) # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


    def load_data(self,data_folder, data_name, label_name):
        with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        return (x_train, y_train)


def get_data(BATCH_SIZE,shuffle_flag):
    My_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1370,0.3081),
    ]  
    )


    trainDataset = DealDataset('./Part 3/data/MNIST/raw', "train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",transform=My_transform)
    testDataset  = DealDataset("./Part 3/data/MNIST/raw", "t10k-images-idx3-ubyte.gz" ,"t10k-labels-idx1-ubyte.gz",transform=My_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=BATCH_SIZE, 
        shuffle=shuffle_flag,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=BATCH_SIZE, 
        shuffle=shuffle_flag,
    )
    
    return train_loader,test_loader


