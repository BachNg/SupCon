import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import ResidualNet, model_short_vgg_contrast, LinearClassifier
from loss import SupContrastLoss
from torchsummary import summary
from tqdm import tqdm

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


train_data = datasets.CIFAR10("./ci_far10_train", download=True, train = True, transform = trans_train)
loader_train = DataLoader(train_data, batch_size=128, shuffle= True)

test_data = datasets.CIFAR10("./ci_far10_test", download=True, train= False, transform= trans_test)
loader_test = DataLoader(test_data, batch_size=128, shuffle= False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model_short_vgg_contrast()
checkpoint = torch.load('./sup_model_3.56.pt')
model.load_state_dict(checkpoint['model'])
model.to(device)


classifier = LinearClassifier()
classifier.to(device)


epoch = 300
optimizer = optim.Adam(classifier.parameters())
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)
criterion = nn.CrossEntropyLoss()

for i in range(epoch):
    epoch_loss = 0
    model.eval()

    classifier.train()
    for data, labels in tqdm(loader_train):
        data = data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            features = model.encoder(data)
        output = classifier(features.detach())
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = np.round(epoch_loss/len(loader_train), 2)
    print("Epoch {} loss train: {}".format(i+1, epoch_loss/len(loader_train)))
    
    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, label in tqdm(loader_test):
            data = data.to(device)
            label = label.to(device)
            features = model.encoder(data)
            output = classifier(features.detach())
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('Accuracy test: {}%'.format(100 * correct / total))
    lr_scheduler.step()


