import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN,OPTICS
from matplotlib import pyplot as plt
import glob
data_path='../../Data/Morphle/data/slides/uploaded/*/pre-processed/rpi_images/x1y0.jpg'
files = list(glob.glob(data_path))[:1]
import json
from PIL import Image,ImageCms
srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")

k=0
size_scale=1/2
eps=10
size=int(512*size_scale)

use_xy_for_clustering=True
xy_scaling_hyperparam=10*2*2*2/size_scale
normalise=False
use_lab=True
method="dbscan"
min_samples=int(60*size_scale)
from utils import pil_to_lab
with open("../../Output/labels.jsonl","w") as f:
    for file in files:
        k+=1
        img = Image.open(file)
        img=img.resize((size,size))
        _img=np.array(img)
        if use_lab:
            Lab_orig =ImageCms.applyTransform(img, rgb2lab)
            L,a,b=Lab_orig.split()
            l_numpy=np.array(L)
            a_numpy=np.array(a)
            b_numpy=np.array(b)
            _numpy_lab=np.stack((l_numpy,a_numpy,b_numpy),axis=-1)
            _img_normalised=pil_to_lab(_numpy_lab,channel_last=True)
        else:
          _img_normalised  =(np.array(img)-128)/128
        
        if use_xy_for_clustering:
            x_indexes=np.broadcast_to((xy_scaling_hyperparam*(np.arange(_img_normalised.shape[0])-_img_normalised.shape[0]/2)/_img_normalised.shape[0]).reshape(_img_normalised.shape[0],1,1),(*_img_normalised.shape[:2],1))
            y_indexes=np.broadcast_to((xy_scaling_hyperparam*(np.arange(_img_normalised.shape[1])-_img_normalised.shape[1]/2)/_img_normalised.shape[1]).reshape(1,_img_normalised.shape[1],1),(*_img_normalised.shape[:2],1))
            _img_indexed=np.concatenate([x_indexes,y_indexes,_img_normalised],axis=-1)
        else:
            _img_indexed=_img_normalised

        if normalise:
            _img_indexed=(_img_indexed-np.mean(_img_indexed,axis=(0,1)))/np.std(_img_indexed,axis=(0,1))
        # print(img.shape)
        # _img=np.array(img)
        # _img=_img/255
        # print(_img.shape)
        _img_list=_img_indexed.reshape(size*size,_img_indexed.shape[-1])

        if method=="dbscan":
            clustering = DBSCAN(eps=eps,min_samples=5).fit(_img_list)
        else:
            clustering = OPTICS(min_samples=50).fit(_img_list)
        labels=clustering.labels_
        
        clusters=list(set(labels))
        print(clusters)
        mask=np.ones(_img_list.shape[:1])
        im = Image.fromarray(_img.astype(np.uint8)).resize((512,512))
        im.save("../../Output/img_%03d.png"%k)
        
        for i,label in enumerate(clusters):
            mask=np.zeros(_img_list.shape[:1])
            mask[(labels==label).nonzero()]=1
            mask=mask.reshape((*_img.shape[:2],1))
            im = Image.fromarray(_img*mask.astype(np.uint8)).resize((512,512))
            im.save("../../Output/img_%03d_mask_%02d.png"%(k,i))
        
        json.dump({
            "clusters":clusters,
            "labels":labels
        },f,default=str)
        f.write("\n")
        print(k,file,set(labels))
    
