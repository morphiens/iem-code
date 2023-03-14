import os
import numpy as np
from scipy import io
from PIL import Image,ImageCms
srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
from utils import save_image,lab_to_pil,pil_to_lab
import torch
import torchvision.transforms as transforms
import glob

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,dataPath,transform=transforms.ToTensor()) -> None:
        super().__init__()
        self.files=list(glob.glob(dataPath))
        self.transform=transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path=self.files[idx]
        img = self.transform(Image.open(img_path).convert('RGB'))
        seg=img.movedim(0,-1)
        seg=(seg[:1,:,:]<1).numpy()
        seg = (seg * 255).astype('uint8').repeat(3,axis=0)
        seg = self.transform(Image.fromarray(seg))[:1]
        return img * 2 - 1, seg
    

class LabDataset(torch.utils.data.Dataset):
    def __init__(self,rgb_dataset) -> None:
        super().__init__()
        self.rgb_datset:torch.utils.data.Dataset=rgb_dataset
        self.pilTransformer=transforms.ToPILImage()
    def __len__(self):
        return self.rgb_datset.__len__()
    def __getitem__(self,idx):
        _tuple=self.rgb_datset.__getitem__(idx)
        if len(_tuple)==2:
            img,seg=_tuple
        else:
            img,seg,_meta=_tuple
        img=self.pilTransformer((img+1)/2)
        Lab_orig =ImageCms.applyTransform(img, rgb2lab)
        L,a,b=Lab_orig.split()
        l_numpy=np.array(L)
        a_numpy=np.array(a)
        b_numpy=np.array(b)
        _numpy_lab=np.stack((l_numpy,a_numpy,b_numpy),axis=-1)
        img=torch.tensor(_numpy_lab,dtype=torch.uint8).movedim(-1,0)
        if len(_tuple)==2:
            return pil_to_lab(img),seg
        else:
            return pil_to_lab(img),seg,_meta
        

    
    

class MorphleDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(MorphleDataset, self).__init__()
        files = list(glob.glob(dataPath))#[40:41]
        self.files =list(filter(lambda p:os.path.exists(self.get_mask_path(p)),files))
        # self.files=self.files[55:56]
        self.transform = transform
        self.datapath = dataPath
    def __len__(self):
        return len(self.files)
    
    
    @staticmethod
    def get_img_path(item):
        # _root=item[:item.index("/pre-processed")]
        # img_path = "%s/pre-processed/rpi_images/x1y0.jpg" % _root
        return item
    @staticmethod
    def get_mask_path(item):
        _root=item[:item.index("/pre-processed")]
        seg_path = "%s/pre-processed/overlayed_mask_for_focus_level_0.jpg" % _root
        return seg_path
    def __getitem__(self, idx):
        item=self.files[idx]
        _root=item[:item.index("/pre-processed")]
        # img_path = "%s/pre-processed/rpi_images/x1y0.jpg" % _root
        img_path=self.files[idx]
        seg_path = "%s/pre-processed/overlayed_mask_for_focus_level_0.jpg" % _root
        img = self.transform(Image.open(img_path))
        
        
        seg=img*0
        if os.path.exists(seg_path):
            seg = np.array(Image.open(seg_path))
            seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 0))
            seg = (seg * 255).astype('uint8').repeat(3,axis=2)
            seg = self.transform(Image.fromarray(seg))[:1]
        return img * 2 - 1, seg,_root+"_"+os.path.basename(img_path)

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