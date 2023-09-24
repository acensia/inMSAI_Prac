import torch
import torch.nn as nn
import torchvision.models as models

class VGG11(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG11,self).__init__()
        self.features = models.vgg11(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
        
# model = VGG11()
# torch.save(model.state_dict(), "./vgg11-bbd30ac9.pth")
