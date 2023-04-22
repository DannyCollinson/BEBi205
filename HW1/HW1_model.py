import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Data should be in './keren'


class HW1Model():
    def __init__(self):
        self.meta = 0
        with open('./keren/meta.yaml', 'r') as stream:
            self.meta = yaml.safe_load(stream)       
        self.cells = []
        self.cell_areas = []
        self.types = []
        self.good_ctypes = [i for i in range(17) if i not in [0,1,17]]
        self.good_channels = [i for i in range(51) if i not in [0,3,21,23,35,37,42,43,48]]
        self.cell_sums = np.zeros((0, len(self.good_channels)))
        self.marker_panel = np.zeros((len(self.good_ctypes), len(self.good_channels)))
        self.marker_panel_count = np.zeros((len(self.good_ctypes), len(self.good_channels)))
        self.types_1hot = None
        self.batch_size = 8
        self.dataset = None
        self.train_loader = None
        self.test_loader = None
        self.model = None

    
    def histogram_equalization(image, number_bins=256):
        histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
        cdf = histogram.cumsum()
        cdf = (number_bins-1) * cdf / cdf[-1]
        equalized = np.interp(image.flatten(), bins[:-1], cdf)
        return equalized.reshape(image.shape)
    
    def parse_data(self, lim=20):
        for root, subdir, files in os.walk('./keren'):
            cur = 0
            for fname in files[1:]:
                if lim > cur:
                    with np.load(os.path.join(root,fname), allow_pickle=True) as f:
                        Xf = np.asarray(f['X'])
                        Xfnorm = np.zeros_like(Xf).squeeze()
                        for channel in self.good_channels:
                            Xfnorm[:,:,channel] = histogram_equalization(Xf[0,:,:,channel])
                        yf = np.asarray(f['y'])
                        ctypes = f['cell_types'].tolist()
                        for cell in range(yf[0,:,:,1].max()):
                            if ctypes[cell] not in self.good_ctypes:
                                continue
                            cell_bin = np.argwhere(yf[0,:,:,1] == cell)
                            cell_binary = (yf[0,:,:,1] == cell)
                            cell_area = np.sum(cell_binary)
                            xmin,ymin = cell_bin.min(axis=0)
                            xmax,ymax = cell_bin.max(axis=0)
                            cview = Xfnorm[xmin:xmax+1, ymin:ymax+1, self.good_channels]
                          # cells.append(cview)
                            self.cell_areas.append(cell_area)
                            self.types.append(ctypes[cell])
                            ch_sums = np.zeros((1,len(self.good_channels)))
                            for ch_ind in range(len(self.good_channels)):
                                ch_sums[0, ch_ind] = (cview[:,:,ch_ind] *\
                                                   (cell_binary[xmin:xmax+1, ymin:ymax+1])).sum()
                                self.marker_panel[ctypes[cell]-2, ch_ind] += ch_sums[0, ch_ind]/cell_area
                                self.marker_panel_count[ctypes[cell]-2, ch_ind] += 1
                            self.cell_sums = np.concatenate((self.cell_sums, ch_sums))
                    cur += 1
        self.marker_panel_count = np.maximum(self.marker_panel_count, np.ones_like(self.marker_panel_count))
        self.marker_panel = self.marker_panel / self.marker_panel_count
    
    class CellMarkerDataset(Dataset):
        def __init__(self, labels, markers, transform=None, target_transform=None):
            self.labels = labels
            self.marker_data = markers
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return torch.from_numpy(self.marker_data[idx,:]), self.labels[idx]
    
    def data_prep(self):
        self.types_1hot = np.zeros((len(self.types),17))
        for i, ty in enumerate(self.types):
            self.types_1hot[i,ty] = 1
        self.dataset = CellMarkerDataset(self.types_1hot, self.cell_sums/((self.cell_sums.max(axis=1).reshape((92901,1)))))
        train_test_extra = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
        self.train_loader = DataLoader(train_test_extra[0], batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(train_test_extra[1], batch_size=self.batch_size, shuffle=False)
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(len(good_channels), 34)
            self.lin2 = nn.Linear(34, 27)
            self.lin3 = nn.Linear(27, 21)
            self.lin4 = nn.Linear(21, 17)

        def forward(self, x):
            x = torch.Tensor(x.type(torch.float))
            x = F.relu(F.dropout(self.lin1(x)))
            x = F.relu(F.dropout(self.lin2(x)))
            x = F.relu(F.dropout(self.lin3(x)))
            x = self.lin4(x)
            return F.softmax(x, dim=1)
    
    def load_model(self, PATH):
        self.model = Net()
        self.model.load_state_dict(torch.load(PATH))
   
    def train(self):
        net = Net()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        for epoch in range(20):
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 4:
                print('Epoch:', epoch+1)

        print('Finished Training')
    
    def evaluate(self):
        correct = 0
        total = 0
        preds = []
        truth = []
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                _, true_labels = torch.max(labels, 1)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                preds.extend(predicted)
                truth.extend(true_labels)
                total += true_labels.size(0)
                correct += (predicted == true_labels).sum().item()
        return 100*correct//total, preds, truth
    
    def predict(self, X, y):
        cell_sums = np.zeros((0, len(self.good_channels)))
        Xnorm = np.zeros_like(X).squeeze()
        for channel in self.good_channels:
            Xnorm[:,:,channel] = histogram_equalization(X[0,:,:,channel])
        for cell in range(y[0,:,:,1].max()):
            cell_bin = np.argwhere(y[0,:,:,1] == cell)
            cell_binary = (y[0,:,:,1] == cell)
            cell_area = np.sum(cell_binary)
            xmin,ymin = cell_bin.min(axis=0)
            xmax,ymax = cell_bin.max(axis=0)
            cview = Xnorm[xmin:xmax+1, ymin:ymax+1, self.good_channels]
          # cells.append(cview)
            ch_sums = np.zeros((1,len(self.good_channels)))
            for ch_ind in range(len(self.good_channels)):
                ch_sums[0, ch_ind] = (cview[:,:,ch_ind] *\
                                   (cell_binary[xmin:xmax+1, ymin:ymax+1])).sum()
            cell_sums = np.concatenate((cell_sums, ch_sums))
        load_model('./HW1_net.pth')
        dataset = CellMarkerDataset(None, cell_sums/((cell_sums.max(axis=1, keepdims=True))))
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                preds.extend(predicted)
        predictionary = {}
        for i, pred in enumerate(preds):
            predictionary[i] = pred
        return preds