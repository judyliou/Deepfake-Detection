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
from utils import VideoDatasetArray, FineTune, freeze_until, cnn_model

image_size = 224 #299
batch_size = 64
epoch = 10
n_frames = 10
hidden_dim = 100
log_name = 'lstm_100_0.0001SGD'

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

class LSTM(nn.Module):
    '''
    Ensemble all results from different frames and train the last layer as a classifier.
    '''
    def __init__(self, feature_extracter, n_frames, hidden_dim, maxpool=False):
        super(LSTM, self).__init__()
        self.feature_extracter = feature_extracter
        self.lstm = nn.LSTM(2048, hidden_dim)
        self.maxpool = nn.MaxPool1d(n_frames)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.mp = maxpool        
    
    def forward(self, images):
        batch_size = images[0].shape[0]
        
        X = []
        for x in images:
            x = torch.squeeze(self.feature_extracter(x), dim=3)
            x = torch.transpose(x, 1, 2)
            X.append(x)
        features = torch.transpose(torch.cat(X, dim=1), 0, 1)
        output, (h, _)= self.lstm(features)  # (len, batch, hidden_dim)
        if not self.mp:
            x = self.classifier(h.view(batch_size, -1))
            x = self.sigmoid(x)
        else:
            output = self.maxpool(output.transpose(0,2)).transpose(0,2).view(batch_size, -1)
            x = self.classifier(output)
            x = self.sigmoid(x)
        return x       

class GRU(nn.Module):
    '''
    Ensemble all results from different frames and train the last layer as a classifier.
    '''
    def __init__(self, feature_extracter, n_frames, hidden_dim1, hidden_dim2):
        super(GRU, self).__init__()
        self.feature_extracter = feature_extracter
        self.gru1 = nn.GRU(2048, hidden_dim1)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2)
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images):
        X = []
        for x in images:
            x = torch.squeeze(self.feature_extracter(x), dim=3)
            x = torch.transpose(x, 1, 2)
            X.append(x)
        features = torch.transpose(torch.cat(X, dim=1), 0, 1)
        output, _ = self.gru1(features)
        output = self.drop(output)
        _, h = self.gru2(output)
        h = self.drop(h)
        x = self.classifier(h)
        x = self.sigmoid(x)
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

    # Train model
    conv_model = cnn_model()
    model = LSTM(conv_model, n_frames=n_frames, hidden_dim=50, maxpool=True)
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    finetune = FineTune(model, 'lstm', epoch=epoch, batch_size=batch_size, optimizer=optimizer, 
                        filename=log_name, trainset_loader=trainset_loader, valset_loader=valset_loader, device=device)
    finetune.train()