from PIL import Image,ImageCms
import torch
import numpy as np
srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
lab2rgb=ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")
def pil_to_lab(_tensor,dtype=torch.float32,channel_last=False):
    

    if not isinstance(_tensor,np.ndarray):
        if channel_last:
            _tensor=_tensor.move(-1,0)
        l=_tensor[0,:,:].to(torch.float32)*(100/255)
        a=_tensor[1,:,:].to(torch.float32)-128
        b=_tensor[2,:,:].to(torch.float32)-128
        return torch.stack([l,a,b],axis=-1 if channel_last else 0)
    else:
        if channel_last:
            _tensor=np.moveaxis(_tensor, -1, 0)
        l=_tensor[0,:,:].astype(np.float32)*(100/255)
        a=_tensor[1,:,:].astype(np.float32)-128
        b=_tensor[2,:,:].astype(np.float32)-128
        return np.stack([l,a,b],axis=-1 if channel_last else 0)
def lab_to_pil(_tensor,channel_last=False):
    
    l=_tensor[0,:,:]*(255/100)
    a=_tensor[1,:,:]+128
    b=_tensor[2,:,:]+128
    if not isinstance(_tensor,np.ndarray):
        return torch.stack([l,a,b],axis=-1 if channel_last else 0)
    else:
        return np.stack([l,a,b],axis=-1 if channel_last else 0)
def save_image(lab,save_file):
    _tensor=lab_to_pil(lab).movedim(0,-1)
    
    Lab=Image.fromarray(_tensor.detach().cpu().numpy().astype(np.uint8))
    
    roundTrip=ImageCms.applyTransform(Lab,lab2rgb)
    roundTrip.save(save_file)

