import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class NatPatchDataset(Dataset):

    def __init__(self, N:int, width:int, height:int, border:int=4, fpath:str='./data/IMAGES.mat'):
        super(NatPatchDataset, self).__init__()
        self.N = N
        self.width = width
        self.height = height
        self.border = border
        self.fpath = fpath
        # holder
        self.images = None
        # initialize patches
        self.extract_patches_()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

    def extract_patches_(self):
        # load mat
        X = loadmat(self.fpath)
        X = X['IMAGES']
        img_size = X.shape[0]
        n_img = X.shape[2]
        self.images = torch.zeros((self.N * n_img, self.width, self.height))
        # for every image
        counter = 0
        for i in range(n_img):
            img = X[:, :, i]
            # plt.figure
            # plt.imshow(img)
            # plt.show()
            
            # noise = torch.randn([img_size,img_size]) * 0.2
            # noisy_img = torch.Tensor(img) + noise
            # img = torch.clamp(noisy_img, 0, 1)
            
            # plt.figure
            # plt.imshow(img)
            # plt.show()
            # plt.figure
            for j in range(self.N):
                # x = np.random.randint(self.border, img_size - self.width - self.border)
                # y = np.random.randint(self.border, img_size - self.height - self.border)
                x = j%int(self.N/self.width)*self.width%int(img_size)
                y = int(j/int(img_size/self.height))*self.height
                # print(x,y)
                crop = torch.tensor(img[x:x+self.width, y:y+self.height])
                self.images[counter, :, :] = crop - crop.mean()
                counter += 1
                # plt.subplot(32,32,j+1)
            #     plt.imshow(crop)
            # plt.show()
