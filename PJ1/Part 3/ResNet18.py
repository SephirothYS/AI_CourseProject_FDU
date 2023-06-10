import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch

class ResNet18(nn.Module):
    def __init__(self,input_ch,classes):
        super(ResNet18,self).__init__()
        self.model = models.resnet18(pretrained = True)
        conv1 = nn.Conv2d(input_ch,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.model.conv1 = conv1
        self.model.fc = nn.Linear(512,classes)
        
    def forward(self,x):
        x = self.model(x)
        return x
    
def train(model,LR,lr_decay_step,MAX_EPOCH,train_loader,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
    model.to(device)
    
    log_interval = 10
    val_interval = 1
    for epoch in range(0, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        for i, data in enumerate(train_loader):


            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)


            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()


            loss_mean += loss.item()
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.



        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(test_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val/len(test_loader)

                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, MAX_EPOCH, j+1, len(test_loader), loss_val_mean, correct_val / total_val))
                
def test(model,test_loader):
    correct_val = 0.
    total_val = 0.
    loss_val = 0.
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

            loss_val += loss.item()

        loss_val_mean = loss_val/len(test_loader)

        print("Valid:\t Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
             j+1, len(test_loader), loss_val_mean, correct_val / total_val))
