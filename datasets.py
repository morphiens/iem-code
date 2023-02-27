import os
import numpy as np
from scipy import io
from PIL import Image
import torch
import torchvision.transforms as transforms
import glob

class MorphleDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(MorphleDataset, self).__init__()
        self.files = list(glob.glob(dataPath)) 
        self.transform = transform
        self.datapath = dataPath
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img_path = "%s/pre-processed/uncropped.jpg" % self.files[idx]
        seg_path = "%s/pre-processed/overlayed_mask_for_focus_level_0.jpg" % self.files[idx]
        img = self.transform(Image.open(img_path))
        
        
        seg=img*0
        if os.path.exists(seg_path):
            seg = np.array(Image.open(seg_path))
            seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 0))
            seg = (seg * 255).astype('uint8').repeat(3,axis=2)
            seg = self.transform(Image.fromarray(seg))[:1]
        return img * 2 - 1, seg
class FlowersDataset(torch.utils.data.Dataset):
    # from https://github.com/mickaelChen/ReDO/blob/master/datasets.py
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(FlowersDataset, self).__init__()
        self.files =  io.loadmat(os.path.join(dataPath, "setid.mat"))
        if sets == 'train':
            self.files = self.files.get('tstid')[0]
        elif sets == 'val':
            self.files = self.files.get('valid')[0]
        else:
            self.files = self.files.get('trnid')[0]
        self.transform = transform
        self.datapath = dataPath
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        imgname = "image_%05d.jpg" % self.files[idx]
        segname = "segmim_%05d.jpg" % self.files[idx]
        img = self.transform(Image.open(os.path.join(self.datapath, "jpg", imgname)))
        seg_path=os.path.join(self.datapath, "segmim", segname)
        seg=img*0
        if os.path.exists(seg_path):
            seg = np.array(Image.open(seg_path))
            seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))
            seg = (seg * 255).astype('uint8').repeat(3,axis=2)
            seg = self.transform(Image.fromarray(seg))[:1]
        return img * 2 - 1, seg