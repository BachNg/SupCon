from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLayer(nn.Module):
  def __init__(self):
    super(IdentityLayer, self).__init__()
  def forward(self, x):
    return F.normalize(x, dim =1)

class ResidualNet(ResNet):
  def __init__(self):
    super(ResidualNet,self).__init__(BasicBlock, [2,2,2,2])
    # self.conv1 = nn.Conv2d(3, 64, 7, stride =1, padding = 3, bias = False)
    self.avgpool = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=(1,1)),
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 128)
    )
    self.fc = IdentityLayer()

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.block_conv1 = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding= 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, padding= 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2) #16
    ) #64 x 16 x 16
    self.block_conv2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2) #8
    ) #128 x 8 x 8
    self.block_conv3 = nn.Sequential(
        nn.Conv2d(128, 256, 3, padding = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding = 1),
        nn.BatchNorm2d(256),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(256, 256, 3, padding = 1),
        # nn.BatchNorm2d(256),
        # nn.ReLU(),
        nn.MaxPool2d(2, 2) #4
    ) #256 x 4 x 4
    self.fc_1 = nn.Linear(256*4*4, 512)
    self.fc_2 = nn.Linear(512, 360)
  def forward(self, input):
    x = self.block_conv1(input) #64 x 16 x 16
    x = self.block_conv2(x) #128 x 8 x 8 
    x = self.block_conv3(x) #256 x 4 x 4
    x = x.view(x.size(0), -1)
    # x = self.drop_out(x)
    x = F.relu(self.fc_1(x))
    x = F.relu(self.fc_2(x))
    return x

class model_short_vgg_contrast(nn.Module):
  def __init__(self):
    super(model_short_vgg_contrast, self).__init__()
    self.encoder = Encoder()
    self.fc = nn.Sequential(
        nn.Linear(360, 360),
        nn.ReLU(),
        nn.Linear(360, 128)
    )
  def forward(self, input):
    x = self.encoder(input)
    x = F.normalize(self.fc(x), dim = 1 )
    return x

class LinearClassifier(nn.Module):
  def __init__(self):
    super(LinearClassifier, self).__init__()
    self.fc = nn.Linear(360, 10)
  def forward(self, x):
    return self.fc(x)
