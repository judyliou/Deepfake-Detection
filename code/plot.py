import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class Parse():
    def __init__(self, file_name, model_name):
        self.output_folder = '/content/drive/My Drive/kaggle/output'
        self.filepath = os.path.join(self.output_folder, file_name)
        self.model_name = model_name

        result = np.load(self.filepath)
        self.train_loss = result[0]
        self.train_accuracy = result[1]
        self.val_loss = result[2]
        self.val_accuracy = result[3]

    def plot_loss(self):
        plt.plot(self.train_loss, label=self.model_name + ' training loss')
        plt.plot(self.val_loss, label=self.model_name + ' val loss')
        plt.legend(frameon=False)
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.train_accuracy, label=self.model_name + ' training accuracy')
        plt.plot(self.val_accuracy, label=self.model_name + ' val accuracy')
        plt.legend(frameon=False)
        plt.show()

    def set_titles(self, titles):
        self.titles = titles
    
    def plot_all(self):
        fig, ax = plt.subplots(1, 2, sharex=True, figsize=(17, 5))
        ax[0].plot(self.train_loss, label=self.model_name + ' training loss')
        ax[0].plot(self.val_loss, label=self.model_name + ' val loss')
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("BCELoss")
        ax[0].legend(frameon=False)

        ax[1].plot(self.train_accuracy, label=self.model_name + ' training accuracy')
        ax[1].plot(self.val_accuracy, label=self.model_name + ' val accuracy')
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy (%)")
        ax[1].legend(frameon=False)

if __name__ == '__main__':
    p1 = Parse('lstm_50_20_0.0001SGD_45ep_noMaxPool.npy', 'LSTM w/o maxpool')
    p1.plot_all()