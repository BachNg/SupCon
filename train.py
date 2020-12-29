import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import ResidualNet, model_short_vgg_contrast
from loss import SupContrastLoss
from torchsummary import summary
from tqdm import tqdm

class DoubleAug:
  def __init__(self, transform):
    self.transform = transform
  def __call__(self, x):
    return [self.transform(x), self.transform(x)]

trans_test= transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trans_train= transforms.Compose(
    [ transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
      transforms.RandomHorizontalFlip(),
      transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_data = datasets.CIFAR10("./ci_far10_train", download=True, train = True, transform = DoubleAug(trans_train))

test_data = datasets.CIFAR10("./ci_far10_test", download=True, train= False, transform= trans_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader_train = DataLoader(train_data, batch_size=128, shuffle= True)
loader_test = DataLoader(test_data, batch_size=128, shuffle= False)

model = model_short_vgg_contrast()
model.to(device)


summary(model, (3, 32, 32))


criterion = SupContrastLoss(0.1)
optimizer = optim.SGD(model.parameters(), lr = 0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

checkpoint = torch.load('./sup_model_3.56.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

epoch = 1500
# best = 86
save_freq = 20
for i in range(epoch):
  epoch_loss = 0
  for data, label in tqdm(loader_train):
    data = torch.cat([data[0], data[1]], dim =0).to(device)
    label = label.to(device)
    features = model(data)
    f1, f2 = torch.split(features, [label.size(0), label.size(0)], 0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss = criterion(features, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  logs = np.round(epoch_loss/len(loader_train), 2)
  if i % save_freq:
    torch.save({'model':model.state_dict(),
                'epoch': i,
                'optimizer': optimizer.state_dict() }, "sup_model_{}.pt".format(logs))
  print("Epoch {} loss train: {}".format(i+1, epoch_loss/len(loader_train)))
  
  scheduler.step()



