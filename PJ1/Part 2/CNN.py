import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(                                  #--> (1,28,28)
                in_channels=1,                          #2维卷积层
                out_channels=16,    
                kernel_size=5,      
                stride=1,           
                padding=2,          
            ),                                          # --> (16,28,28)
            nn.ReLU(),                                  #非线性激活层
            nn.MaxPool2d(kernel_size=2),                # --> (16,14,14)
        )
 
        self.conv2=nn.Sequential(
            nn.Conv2d(                                  # --> (16,14,14)
                in_channels=16,     
                out_channels=32,    
                kernel_size=5,      
                stride=1,           
                padding=2,          
            ),                                          # --> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # --> (32,7,7)
        )
 
        self.out=nn.Linear(32*7*7,12)                   #输入全连接层的数据
 
    def forward(self,x):
        x = x.float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x=self.conv1(x)
        x=self.conv2(x)                                 #（batch,32,7,7）
        x=x.view(x.size(0),-1)                          #(batch ,32 * 7 * 7)
        output=self.out(x)
        return output


def train(model,train_loader,test_loader,EPOCH,LR):
    # optimizer=torch.optim.Adam(model.parameters(),lr=LR)    #优化器为Adam
    optimizer=torch.optim.SGD(model.parameters(),lr=LR)    #优化器为SGD
    loss_fn=nn.CrossEntropyLoss()                       #损失函数为交叉熵  
    step=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(EPOCH):
        for step,data in enumerate(train_loader):
            x,y=data

            x = x.to(device)
            y = y.to(device)
            
            output=model(x)         
            loss=loss_fn(output,y.long())
            optimizer.zero_grad()   
            loss.backward()         
            optimizer.step()        
    

            if (step%50==0):

                for s,data_test in enumerate(test_loader):
                    test_x,test_y = data_test
                    test_x , test_y = test_x.to(device) , test_y.to(device)
                    test_output=model(test_x)
                    y_pred=torch.max(test_output,1)[1].data.squeeze()
                    accuracy=sum(y_pred==test_y).item()/test_y.size(0)
        
                    print('now epoch :  ', epoch, '   |  loss : %.4f ' % loss.item(), '     |   accuracy :   ' , accuracy)
                    
def test(model,test_loader):
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for s,data_test in enumerate(test_loader):
        test_x,test_y = data_test
        test_x , test_y = test_x.to(device) , test_y.to(device)
        test_output=model(test_x)
        y_pred=torch.max(test_output,1)[1].data.squeeze()
        count += sum(y_pred==test_y).item()
    accuracy = count / test_loader.__len__()
    print('accuracy :   ' , accuracy)