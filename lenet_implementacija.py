import torch
from torch import nn,optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets,transforms
import torch.nn.functional as F

device =("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])
train_set = datasets.FashionMNIST('DATA_MIST/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64,shuffle=True)

test_set = datasets.FashionMNIST('DATA_MIST/',download=True, train= False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set,batch_size=64, shuffle=True)

train_data_size = len(train_set)
test_data_size = len(test_set)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        self.convolutinal_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,kernel_size=(5,5),stride=(1,1),padding=(0,0)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2),stride=2),
            nn.Conv2d(in_channels=6, out_channels=16,kernel_size=(5,5),stride=(1,1),padding=(0,0)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2),stride=2),
            nn.Conv2d(in_channels=16, out_channels=120,kernel_size=(5,5),stride=(1,1),padding=(0,0)),
            nn.ReLU(),
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
        

    def forward(self,x):
        x = self.convolutinal_layer(x)
        x = torch.flatten(x,1)
        x = self.linear_layer(x)
        #x = F.softmax(x,dim=1)
        return x

x = torch.randn(64,1,32,32)
model = LeNet().to(device)
print(model)

print("Trenirat ću model na ",device)
print()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
criterion = nn.CrossEntropyLoss()

epochs = 20
train_loss, val_loss = [] ,[]

for epoch in range(epochs):
    total_train_loss = 0
    total_val_loss = 0
    
    if(epoch==0): #POčetni ispis evaluacije
        model.eval()
        total = 0
        for idx, (image,label) in enumerate(testloader):
            image, label = image.to(device), label.to(device)
            pred = model(image)
    
            loss = criterion(pred,label)
            total_val_loss += loss.item()
    
            pred = torch.nn.functional.softmax(pred,dim=1)
            for i,p in enumerate(pred):
                if label[i] == torch.max(p.data,0)[1]:
                    total += 1
        accuracy = total / test_data_size
        total_val_loss = total_val_loss / (idx+1)
        train_loss.append(0)
        val_loss.append(total_val_loss)
    
        print(f"Netrenirani model | Train loss: {total_train_loss} | Validation loss: {total_val_loss} | Accuracy: {accuracy}")
    
    model.train()
    for idx, (image,label) in enumerate(trainloader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(image)

        loss = criterion(pred,label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    total_train_loss = total_train_loss / (idx+1)
    train_loss.append(total_train_loss)

    model.eval()
    total = 0
    for idx, (image,label) in enumerate(testloader):
        image, label = image.to(device), label.to(device)
        pred = model(image)

        loss = criterion(pred,label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred,dim=1)
        for i,p in enumerate(pred):
            if label[i] == torch.max(p.data,0)[1]:
                total += 1
    accuracy = total / test_data_size
    total_val_loss = total_val_loss / (idx+1)
    val_loss.append(total_val_loss)

    print(f"Epoch: {epoch} | Train loss: {total_train_loss} | Validation loss: {total_val_loss} | Accuracy: {accuracy}")

plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['Training','Validation'])

testiter = iter(testloader)
images, labels = next(testiter)

with torch.no_grad():
    images, labels = images.to(device), labels.to(device)
    pred = model(images)

images_np = [i.mean(dim=0).cpu().numpy() for i in images]

class_names = ['T-shirt/top', 'Trouser','Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker','Bag','Ankle boot']
fig = plt.figure(figsize=(15,7))
for i in range(50):
    ax = fig.add_subplot(5, 10, i+1, xticks=[], yticks=[])
    ax.imshow(images_np[i], cmap=plt.cm.gray_r, interpolation='nearest')

    if labels[i] == torch.max(pred[i],0)[1]:
        ax.text(0,3,class_names[torch.max(pred[i],0)[1]],color='blue')
    else:
        ax.text(0,3,class_names[torch.max(pred[i],0)[1]],color='red')
plt.tight_layout()
plt.show()

