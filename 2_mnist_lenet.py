import time

import torch 
import torchvision

from typing import List, NewType

from torch import nn, optim
from torch.nn import functional as F 

from utils import loss_curve

Tensor = NewType("Tensor", torch.tensor)
DataLoader = NewType("DataLoader", torch.utils.data.dataloader.DataLoader)
Device = NewType("Device", torch.device)

def mnist_loader(train: bool, batch_size: int = 512) -> DataLoader:
    data_path = "~/data/mnist_data"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                   ])
    datasets = torchvision.datasets.MNIST(data_path, train=train, download=True,
                                        transform=transform)
    data_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=train)
    return data_loader

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        h1 = F.relu(self.conv1(x))
        h2 = F.max_pool2d(h1, 2)
        h3 = F.relu(self.conv2(h2))
        h4 = F.max_pool2d(h3, 2)
        h5 = F.relu(self.conv3(h4))
        h5 = h5.view(h5.size(0), -1)
        h6 = F.relu(self.fc1(h5))
        h7 = F.relu(self.fc2(h6))

        return h7

    def train(self, epochs: int, train_loader: DataLoader, device: Device, lr: float = 1e-3, valid_loader: DataLoader = None) -> List[float]:
        losses = list()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        for epoch in range(epochs):
            start = time.time()
            for idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                if idx%100==99:
                    print("At epoch %d step %d, the loss is %f."%(epoch+1, idx+1, loss.item()))

            end = time.time()
            print("__________ Epoch %.2d __________"%(epoch+1))
            print("The epoch %d cost %.3f s."%(epoch+1, end-start))
            acc = self.valid(train_loader, device)
            print("After epoch %d, the acc on training set is %f."%(epoch+1, acc))
            if valid_loader:
                acc = self.valid(valid_loader, device)
                print("After epoch %d, the acc on validation set is %f."%(epoch+1, acc))
        
        return losses

    def valid(self, valid_loader: DataLoader, device: Device) -> None:
        with torch.no_grad():
            correct = 0
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self(batch_x)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds==batch_y).sum().item()

            return correct / len(valid_loader.dataset)

    def predict(self, test_loader: DataLoader, device: Device) -> Tensor:
        with torch.no_grad():
            pred_lst = list()
            for batch_x in test_loader:
                batch_x = batch_x.to(device)
                outputs = self(batch_x)
                pred_lst.append(outputs.argmax(dim=1))

        return torch.cat(pred_lst)



if __name__ == "__main__":
    train_loader = mnist_loader(train=True, batch_size=1024)
    valid_loader = mnist_loader(train=False, batch_size=1024)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = LeNet().to(device)
    losses = model.train(30, train_loader, device, valid_loader=valid_loader)
    loss_curve(losses)
