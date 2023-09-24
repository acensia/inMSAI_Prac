import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

class RBM(nn.Module):
    def __init__(self, visible_size, hidden_size):
        super(RBM, self).__init__()
        self.W = nn.Parameter(
            torch.randn(visible_size, hidden_size)
        )
        self.v_bias = nn.Parameter(
            torch.randn(visible_size)
        )
        self.h_bias = nn.Parameter(
            torch.randn(hidden_size)
        )
        
        
    def forward(self, x):
        hidden_prob = torch.sigmoid(
            torch.matmul(x, self.W) + self.h_bias
            # xW + b
        )
        hidden_state = torch.bernoulli(
            hidden_prob
        )
        
        visible_prob = torch.sigmoid(
            torch.matmul(hidden_state, torch.transpose(self.W, 0, 1))
            + self.v_bias
        )
        
        return visible_prob, hidden_state
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ),(0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True,transform=transform, download=True
    )
    
    train_loader = DataLoader(train_dataset,batch_size=64, shuffle=True)
    
    visible_size = 28 * 28
    # MNIST dataset img size
    hidden_size = 256
    #hidden layer size, typically 256
    rbm = RBM(visible_size, hidden_size)
    #RBM model
    
    criterion = nn.BCELoss()
    opt = torch.optim.SGD(rbm.parameters(),lr = 0.01)
    
    n_epochs = 10
    for epoch in range(n_epochs):
        for imgs, _ in train_loader:
            inputs = imgs.view(-1, visible_size)
            
            visible_prob, _ = rbm(inputs)
            
            loss = criterion(visible_prob, inputs)
            #compared with ANN -> visible_prob instead of output
            #                  -> inputs instead of labels
            
            opt.zero_grad()
            loss.backward()
            
            opt.step()
        
        print("{}/{} : Loss P{:.4f}".format(epoch+1, n_epochs, loss.item()))
        
        vutils.save_image(rbm.W.view(hidden_size, 1, 28, 28))
        
        inputs_display = inputs.view(-1, 1, 28, 28)
            
    