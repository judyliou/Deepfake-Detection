import os
import glob
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from utils import VideoDatasetArray, FineTune, freeze_until, cnn_model
sys.path.append('/content/drive/My Drive/kaggle/cnn_detection/networks')
from resnet import resnet50

image_size = 224 #299
batch_size = 64
epoch = 10
n_frames = 10
hidden_dim = 100
log_name = 'vgg16_100_0.0001SGD'

# training batch03
data_folder = '/content/drive/My Drive/kaggle/batch03'
metadata_dir = glob.glob(os.path.join(data_folder, 'dfdc_train_part_47', '*.json'))[0]
split = 0.8
model_path = '/content/drive/My Drive/kaggle/cnn_detection'
# valset from batch02
data_folder_val = '/content/drive/My Drive/kaggle/batch02'
metadata_dir_val = glob.glob(os.path.join(data_folder_val, 'dfdc_train_part_48', '*.json'))[0]

transform = {
        'train': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(hue=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]), 
        'val': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        }

class MyEnsemble(nn.Module):
    '''
    Ensemble all results from different frames and train the last layer as a classifier.
    '''
    def __init__(self, pretrained, n_frames):
        super(MyEnsemble, self).__init__()
        self.pretrained = pretrained
        self.classifier = nn.Linear(n_frames, 1)
        self.sigmoid = nn.Sigmoid()
        # self.aux = aux

    def forward(self, images):
        X = []
        AUX = []
        for x in images:
            x = self.pretrained(x)
            if isinstance(x, tuple):
                X.append(x[0])
                AUX.append(x[1])
            else:
                X.append(x)
        x = torch.cat(X, dim=1)
        x = self.sigmoid(self.classifier(x))
        if len(AUX) > 0:
            aux = torch.cat(AUX, dim=1) 
            aux = self.sigmoid(self.classifier(aux))
            return x, aux
        return x   


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")

    # Load data
    print('\n----- Load Training Set -------')
    trainset = VideoDatasetArray(
        root= data_folder, 
        n_frames = n_frames,
        transform=transform, train=True
    )
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    print('\n----- Load Val Set-------')
    valset = VideoDatasetArray(
        root= data_folder_val, 
        n_frames = n_frames,
        transform=transform, train=False
    )
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Define VGG model
    vgg16 = models.vgg16(pretrained=True)
    old_classifier = list(vgg16.classifier.children()) 
    old_classifier = old_classifier[:3]
    old_classifier.append(nn.Linear(4096,1))
    vgg16.classifier = nn.Sequential(*old_classifier) 

    freeze_until(vgg16, "features.17.weight")
    model = MyEnsemble(vgg16, n_frames=n_frames)
    model.cuda()

    # Train VGG model
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    finetune = FineTune(model, 'vgg16', epoch=epoch, batch_size=batch_size, optimizer=optimizer, 
                        filename=log_name, trainset_loader=trainset_loader, valset_loader=valset_loader, device=device)
    finetune.train()

    # Define ResNet Model
    model_path = '/content/drive/My Drive/kaggle/cnn_detection'
    model_file = 'blur_jpg_prob0.1.pth'

    resnet = resnet50(num_classes=1)
    state_dict = torch.load(os.path.join(model_path, model_file), map_location=device)
    resnet.load_state_dict(state_dict['model'])
    
    freeze_until(resnet, "layer4.0.conv1.weight")
    model = MyEnsemble(resnet, n_frames=n_frames)
    model.cuda()

    # Train ResNet model
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    finetune = FineTune(model, 'vgg16', epoch=25, optimizer=optimizer, log_interval=10)
    finetune.train()

    # Define Inception V3 model
    inception = models.inception_v3(pretrained=True) 
    inception.AuxLogits.fc = nn.Linear(768, 1)
    inception.fc = nn.Linear(2048, 1)
    freeze_until(inception, "Mixed_7c.branch_pool.conv.weight")
    model = MyEnsemble(inception, n_frames=n_frames) 
    model.cuda()

    # Train Inception V3 model
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    finetune = FineTune(model, 'vgg16', epoch=25, optimizer=optimizer, filename='0415_inception.txt', log_interval=10)
    finetune.train()