import torch
import torch.nn as nn
import os
import glob
from PIL import Image
import numpy as np
import tqdm
from resnet import resnet50
import torchvision.models as models

class VideoDatasetArray(Dataset):
    def __init__(self, root, n_frames, transform=None, train=True):
        """ Intialize the dataset from npy files
        
        Args:
            - root: root directory of the data
            - n_frame: the number of frames for each video
            - tranform: a custom tranform function
            - train: dataset for training
        """
        self.root = root
        self.transform = transform['train' if train else 'val']
        face_dir = os.path.join(self.root)
        if train:
            face_file = glob.glob(os.path.join(face_dir, '*.npy'))
        else:
            face_dir = os.path.join(self.root, 'face10train')
            face_file = [glob.glob(os.path.join(face_dir, '*.npy'))[1]]

        # Preload dataset to memory
        self.labels = []
        self.images = []
        print("\nPreload dataset to memory...\n")
        for face_batch in tqdm.tqdm(face_file, ncols=80):
            data = np.load(face_batch, allow_pickle=True)
            labels = data.item()['y']
            for k in range(len(labels)):
                target = 1 if labels[k] == "FAKE" else 0
                collections = []
                for i in range(10):
                    image = data.item()['x' + str(i)][k].transpose()
                    collections.append(image.copy())
                self.images.append(collections)
                self.labels.append(target)
            # del data
            
        self.len = len(self.labels)
    
    def __getitem__(self, index):
        images = self.images[index]
        label = self.labels[index]
        X = []
        if self.transform is not None:
            for image in images:
                # image = torch.FloatTensor(image)
                x = Image.fromarray(image.astype(np.uint8).transpose(1,2,0))
                X.append(self.transform(x))
        return X, label
    
    def __len__(self):
        return self.len

class FineTune():
    def __init__(self, model, model_name, epoch, batch_size, optimizer, filename,
                 trainset_loader, valset_loader, device, log_interval=100):
        self.model = model
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.trainset_loader = trainset_loader
        self.valset_loader = valset_loader
        self.device = device

        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.output_folder = '/content/drive/My Drive/kaggle/output'
        self.filename = filename

    def train(self):  # set training mode
        loss_fn = nn.BCELoss()
        for ep in range(self.epoch):
            self.model.train()
            iteration = 0
            for batch_idx, (data, target) in enumerate(self.trainset_loader):
                data = [_data.to(self.device) for _data in data] 
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output.squeeze(dim=1), target.type_as(output))
                loss.backward()
                self.optimizer.step()
                if iteration % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * self.batch_size, len(self.trainset_loader.dataset),
                        100. * (batch_idx+1) / len(self.trainset_loader), loss.item()))
                iteration += 1

            # Evaluation for both the training set and validation set
            self.eval(False)
            self.eval(True)

            # Save
            history = [self.train_loss, self.train_accuracy, self.val_loss, self.val_accuracy]
            np.save(os.path.join(self.output_folder, self.filename), history)
            torch.save({
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                        }, os.path.join(self.output_folder, self.filename+'.pt'))
        
        # Save loss and accuracy
        output_file = os.path.join(self.output_folder, self.filename+'.txt')

    def eval(self, is_val=True):
        loss_fn = nn.BCELoss(reduction="sum")
        self.model.eval()  # set evaluation mode
        loss = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        data_loader = self.valset_loader if is_val else self.trainset_loader
        with torch.no_grad():  # set all requires_grad flags to False
            for data, target in data_loader:
                data = [_data.to(device) for _data in data] 
                target = target.to(device)
                output = self.model(data)
                loss += loss_fn(output.squeeze(dim=1), target.type_as(output)).item()
                pred = (output > 0.5).int()
                correct += pred.eq(target.view_as(pred)).sum().item()

                if is_val:
                    # for calculating precision and recall
                    TP += (pred * target.view_as(pred)).sum().item()
                    TN += ((1 - pred) * (1 - target.view_as(pred))).sum().item()
                    FP += (pred * (1 - target.view_as(pred))).sum().item()
                    FN += ((1 - pred) * target.view_as(pred)).sum().item()

        loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)

        if is_val:
            # save validation loss and accuracy
            self.val_loss.append(loss)
            self.val_accuracy.append(accuracy)

            # calculate precision, recall, and f1
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(
                loss, correct, len(data_loader.dataset),
                accuracy, precision, recall, f1))
        else:
            self.train_loss.append(loss)
            self.train_accuracy.append(accuracy)
            print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                loss, correct, len(data_loader.dataset), accuracy))

class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.resnet_layer(x)
        return x

def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
    
    fine_tuned = [k for k,v in net.named_parameters() if v.requires_grad]
    print('Layer to fine-tune:', fine_tuned)

def cnn_model():
    resnet50 = models.resnet50(pretrained=True)
    pretrained_dict = resnet50.state_dict()
    model = ResNet(resnet50)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for param in model.parameters():
        param.requires_grad = False
    return model
