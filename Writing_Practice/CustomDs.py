import os
from tqdm import tqdm
import glob
from PIL import Image
import re

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from sklearn.model_selection import train_test_split

slash = "\\"

            
def label_parser(filename):
    if filename.find("_") == -1:
        label_num = re.sub(r'[^0-9]','',filename)
        return int(label_num)
    else :
        label_num = filename.split("_")[-1].split(".")[0]
        return int(label_num)

def Check_broken(dir):
    paths = glob.glob(os.path.join(dir, "*.*"))
    _trash = []
    for p in paths:
        try:
            i = Image.open(p)
        except:
            print(f"File {p} is broken")
            _trash.append(p)


class CustomDataset():
    def __init__(self, raw_dir, transform=None, gray=True):
        # self.raw_paths = glob.glob(os.path.join(raw_dir, "*.*"))
        self.img_root = raw_dir
        
        self.raw_paths = os.listdir(raw_dir)
        self.transform = transform
        self.labels = {}
        self.gray = gray
        
    def __getitem__(self, index):
        # raw_path : str = self.raw_paths[index]
        # img = Image.open(raw_path)
        
        raw_path = self.raw_paths[index]
        img = Image.open(os.path.join(self.img_root, 
                                      raw_path)).convert("L")
        
        
            
        if self.gray :
            img = img.convert("L")
            
        if self.gray == (img.mode == "L"):
            filename = raw_path.split(slash)[-1]
            label = label_parser(filename)
            
            if self.transform:
                img = self.transform(img)
            if label not in range(10):
                print("Check your image : ", filename)
                return
            return img, label
        else :
            print(f"{self.gray}, {img.mode}")
            print("Check your dataset mode")
            
    def __len__(self):
        return len(self.raw_paths)




class CNN(nn.Module):
    def __init__(self) :
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # input channel 1 -> gray scale
        # output channel 16 -> arbitrarily  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1,padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
        # 32channels 7*7 mat
        
        """
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        (btwn cv2, fc)
        Why define relu distributely as 1, 2??
        """
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        # after pooling, no more activation func.
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def CNN_activation(train_dataset, test_dataset):
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = CNN().to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    epoch_losses = []
    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    fig.tight_layout(pad=4.0)
    axs = axs.flatten()
    
    for epoch in range(num_epochs):
        model.train()
        
        running_loss = 0.0
        #error value after train loader iteration
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_losses.append(epoch_loss)
        
        
        print("Epoch {} / {}, Loss : {:.4f}".format(epoch+1,num_epochs,epoch_loss))
        
        if epoch == 0:
            # conv. layer weight visualization
            weights = model.conv1.weight.detach().cpu().numpy()
            axs[0].imshow(weights[0,0],cmap='coolwarm')
            axs[0].set_title("Conv1 Weights")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(axs[0].imshow(weights[0,0],cmap='coolwarm'), cax=cax)
            
            weights = model.conv2.weight.detach().cpu().numpy()
            axs[1].imshow(weights[0,0],cmap='coolwarm')
            axs[1].set_title("Conv2 Weights")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(axs[1].imshow(weights[0,0],cmap='coolwarm'), cax=cax)
            
        axs[3].plot(range(epoch+1), epoch_losses)
        axs[3].set_title('training loss')
        axs[3].set_xlabel('Epoch')
        axs[3].set_ylabel('Loss')            
            
    plt.show()
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
            
    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f} %")


if __name__ == "__main__":
    tar_d = "./Letters"
    # Check_broken(tar_d)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    d = CustomDataset(tar_d, transform=transform)
    # for i in d:
    #     print(i)
    
    train_set, test_set = train_test_split(d, test_size=0.2,random_state=777)
    print("Splited as ", len(train_set), len(test_set))
    
    # temp_dict = dict.fromkeys(list(range(10)), 0)
    # for _, label in train_set:
    #     temp_dict[label] += 1

    # print(temp_dict)
    # exit()
    
    CNN_activation(train_set, test_set)