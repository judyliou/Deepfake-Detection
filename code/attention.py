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
log_name = 'attention_100_0.0001SGD'

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
        
class Attention(nn.Module):
    def __init__(self, feature_extracter, n_frames, hidden_size, conv_features=2048, maxpool=False):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.conv_features = conv_features
        self.n_frames = n_frames
        self.positional_encodings = self.create_positional_encodings()
        self.mp = maxpool

        self.feature_extracter = feature_extracter
        self.Q = nn.Linear(conv_features, hidden_size)
        self.K = nn.Linear(conv_features, hidden_size)
        self.V = nn.Linear(conv_features, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(hidden_size, dtype=torch.float))
        self.maxpool = nn.MaxPool1d(n_frames)
        if self.mp:
            self.classifier = nn.Linear(hidden_size, 1)
        else:
            self.classifier = nn.Linear(hidden_size*n_frames, 1)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images):
        batch_size = images[0].shape[0]
        
        X = []
        for x in images: # (batch, 3, 224, 224)
            x = torch.squeeze(self.feature_extracter(x), dim=3)
            x = torch.transpose(x, 1, 2)
            X.append(x)

        features = torch.cat(X, dim=1) # (batch, frames, 2048)
        features = features + self.positional_encodings[:self.n_frames].unsqueeze(0)
        q = self.Q(features)
        k = self.K(features)
        v = self.V(features)    
        unnormalized_attention = torch.bmm(k, q.transpose(2,1)) * self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention)
        if self.mp:
            context = torch.bmm(attention_weights.transpose(2,1), v)
            context = self.maxpool(context.transpose(1,2)).view(batch_size, -1)
        else:
            context = torch.bmm(attention_weights.transpose(2,1), v).view(batch_size, -1) # (batch, hidden*frames)
        x = self.sigmoid(self.classifier(context))
        return x

    def create_positional_encodings(self, max_seq_len=100):
      pos_indices = torch.arange(max_seq_len)[..., None]
      dim_indices = torch.arange(self.conv_features//2)[None, ...]
      exponents = (2*dim_indices).float()/(self.conv_features)
      trig_args = pos_indices / (10000**exponents)
      sin_terms = torch.sin(trig_args)
      cos_terms = torch.cos(trig_args)

      pos_encodings = torch.zeros((max_seq_len, self.conv_features))
      pos_encodings[:, 0::2] = sin_terms
      pos_encodings[:, 1::2] = cos_terms

      pos_encodings = pos_encodings.cuda()

      return pos_encodings


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
    model = Attention(conv_model, n_frames=n_frames, hidden_size=100, maxpool=True)
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    finetune = FineTune(model, 'attention',  epoch=epoch, batch_size=batch_size, optimizer=optimizer, 
                        filename=log_name, trainset_loader=trainset_loader, valset_loader=valset_loader, device=device)
    finetune.train()
