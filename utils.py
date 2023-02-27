from PIL import Image,ImageCms
import torch
srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
lab2rgb=ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")
def pil_to_lab(_tensor):
    l=_tensor[0,:,:].to(torch.float32)*(100/255)
    a=_tensor[1,:,:].to(torch.float32)-128
    b=_tensor[2,:,:].to(torch.float32)-128
    return torch.stack([l,a,b],axis=0)
def lab_to_pil(_tensor):
    l=_tensor[0,:,:]*(255/100)
    a=_tensor[1,:,:]+128
    b=_tensor[2,:,:]+128
    return torch.stack([l,a,b],axis=0).to(torch.uint8)
def save_image(lab,save_file):
    _tensor=lab_to_pil(lab).movedim(0,-1)
    
    Lab=Image.fromarray(_tensor.detach().cpu().numpy())
    
    roundTrip=ImageCms.applyTransform(Lab,lab2rgb)
    roundTrip.save(save_file)

