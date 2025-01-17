import torch, argparse, time
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from metrics import *
from models import *
from datasets import *
from torchvision.utils import save_image as save_rgb_image
from utils import masked_mean

parser = argparse.ArgumentParser(description='Inpainting Error Maximization')
parser.add_argument('data_path', type=str)
parser.add_argument('--dataset-type',type=str,default='morphle')
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--iters', type=int, default=25)
parser.add_argument('--sigma', type=float, default=3)
parser.add_argument('--kernel-size', type=int, default=11)
parser.add_argument('--reps', type=int, default=2)
parser.add_argument('--lmbda', type=float, default=0.00001)
parser.add_argument('--scale-factor', type=int, default=1)
parser.add_argument('--device',  type=str, default='cpu') #cuda can be used alternatively
parser.add_argument('--boundary-loss',action='store_true')
parser.add_argument('--use-lab',action='store_true')
parser.add_argument('--diff-threshold',type=float,default=1.5)
args = parser.parse_args()
transformsList=[
    transforms.Resize(args.size, transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(args.size),
    transforms.ToTensor()
    
]

if args.use_lab:
    from utils import save_image
    
else:
    from torchvision.utils import save_image
    
transform = transforms.Compose(transformsList)

if args.dataset_type=="morphle":
    data = MorphleDataset(args.data_path,"test",transform)
elif args.dataset_type=="test":
    data = TestDataset(args.data_path,transform)
else:
    data = FlowersDataset(args.data_path, 'test', transform)

if args.use_lab:
    data= LabDataset(data)

loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

# naive inpainting module that uses a Gaussian filter to predict values of masked out pixels
inpainter = Inpainter(args.sigma, args.kernel_size, args.reps, args.scale_factor).to(args.device)
# module that gets mask as input and returns its boundary, used to restrict updates only to boundary pixels
boundary = Boundary().to(args.device)

start_time = time.time()
img_index=0
transition_loss_calculator=TransitionLoss()
continuity_loss_calculator=ContinuityLoss()
blur=GuassianBlur(args.sigma, args.kernel_size,args.reps)
divisions=5
with torch.no_grad():
    for batch_idx, (x, seg,_meta) in enumerate(loader):
        # if batch_idx<60:
        #     continue;
        # if batch_idx>60:
        #     break
        print(len(x))
        print("Batch {}/{}".format(batch_idx+1, len(loader)))
        x, seg = x.to(args.device), seg.to(args.device)
        # guass_x=x
        guass_x=blur(x)
        all_masks=[]
        for division in range(divisions-1):
            # initializes a mask for each sample in the mini batch as a centered square
            mask = torch.nn.Parameter(torch.zeros(len(x), 1, x.shape[-2], x.shape[-1]).to(args.device))
            num_mask_regions=10
            import random
            l = args.size//10
            
            for i in range(num_mask_regions):

                

                gap_x=random.randrange(0,x.shape[-2]-l)
                gap_y=random.randrange(0,x.shape[-1]-l)
                
                mask.data[:,:,gap_x:gap_x+l,gap_y:gap_y+l].fill_(1.0)

            for i in range(args.iters):
                background_mask=1-mask
                #todo move to once every division
                for pm in all_masks:
                    mask=mask*pm
                    background_mask=background_mask*pm
                total_mask=(background_mask+mask)    
                foreground = x * mask
                background = x * background_mask
                
                
                update_bool=switch_masks(guass_x*total_mask,mask,guass_x*mask,guass_x*background_mask,background_mask)    
                if torch.sum(update_bool)==0:
                    break
                
                    # grad = mask.grad.data
                    # we only update mask pixels that are in the boundary AND have non-zero gradient
                    # update_bool = boundary(mask) * (grad != 0)
                    # eta=1e-10
                    # update_bool = grad!=0
                    # update_bool= (torch.abs(grad) >eta)
                    
                    # pixels with positive gradients are set to 1 and with negative gradients are set to 0
                mask.data[update_bool] = torch.abs(1-mask)[update_bool]
                # grad.zero_()
                
                # smoothing procedure: we set a pixel to 1 if there are 4 or more 1-valued pixels in its 3x3 neighborhood
                # mask.data = (F.avg_pool2d(mask, 3, 1, 1, divisor_override=1) >= 4).float()
                
                # acc, iou, miou, dice = compute_performance(mask, seg)
                # print("\tIter {:>3}: InpError {:.3f} IoU {:.3f} DICE {:.3f}".format(i, inp_error.mean().item(), iou, dice))
                # print("\tIter {:>3}: InpError {:.3f} IoU {:.3f} DICE {:.3f}".format(i, 0, iou, dice))
                # img_foreground=foreground[0,:,:,:]
                # save_image(img_foreground,"../../Output/img_%03d_foreground_%03d.png"%(img_index,i))
            
            

            background_mask = (1-mask)*total_mask
            image_pixels=args.size*args.size
            foreground_pixels,background_pixels=torch.sum(mask!=0).item()/image_pixels,torch.sum(background_mask!=0).item()/image_pixels
            total_pixels=foreground_pixels+background_pixels
            print("Pixel counts %.2f %.2f %.2f"%(total_pixels,foreground_pixels,background_pixels))
            foreground = x * mask
            background = x * (background_mask)
                

            fg_mean=masked_mean(foreground[:,:,:,:],mask,dim=(-1,-2))
            bg_mean=masked_mean(background[:,:,:,:],background_mask,dim=(-1,-2))
            diff_masked=torch.sqrt(torch.sum(torch.square(fg_mean-bg_mean)))
            print("Iteration",division,diff_masked)
            
            switch=False
            if x.shape[0]!=1:
                    raise NotImplementedError
            #make whiter image as foreground
            if len(all_masks)==0:
                #only use ab channel
                fg_gap=torch.sum(torch.square(fg_mean[:,1:]),dim=(-1))
                bg_gap=torch.sum(torch.square(bg_mean)[:,1:],dim=(-1))
                print("background hue:",bg_gap)
                print("foreground hue:",fg_gap)
                if bg_gap>fg_gap:
                    switch=True
                
            else:
                fg_gap=torch.mean(torch.abs(masked_mean(foreground[:,:,:,:],mask,dim=(-1,-2))-masked_mean(x*(1-total_mask),(1-total_mask),dim=(-1,-2))),axis=-1)
                bg_gap=torch.mean(torch.abs(masked_mean(background[:,:,:,:],background_mask,dim=(-1,-2))-masked_mean(x*(1-total_mask),(1-total_mask),dim=(-1,-2))),axis=-1)
                print("background gap:",bg_gap)
                print("foreground gap:",fg_gap)
                if fg_gap>bg_gap:
                    switch=True
            if switch:
                
                mask=1-mask
                temp=foreground
                foreground=background
                background=temp
                
            else:
                pass
            all_masks.append(1-mask)

            
            for k in range(foreground.shape[0]):
                if division==0:
                    _meta_dir=os.path.basename(_meta[k])+"_"
                    dump_path=os.path.join("../../Data/Output")
                    os.makedirs(dump_path,exist_ok=True)
                    save_image(x[k,:,:,:],os.path.join(dump_path,_meta_dir+".png"))
                img_index=batch_idx*args.batch_size+k
                save_image(x[k,:,:,:]*total_mask[k,:,:,:],os.path.join(dump_path,_meta_dir+"%01d_%0.2f.png"%(division,diff_masked)))
                img_foreground=foreground[k,:,:,:]
                save_image(img_foreground,os.path.join(dump_path,_meta_dir+"foreground_%01d.png"%(division)))
                # save_rgb_image(seg[k,:,:,:],"../../Output/img_%03d_seg.png"%img_index)
                # save_image((pred_foreground*mask)[k,:,:,:],"../../Output/img_%03d_foreground_pred.png"%img_index)
                img_background=background[k,:,:,:]
                save_image(img_background,os.path.join(dump_path,_meta_dir+"background_%01d.png"%(division)))
                _b=boundary(mask[k:k+1,:,:,:]).to(torch.float)[0,:,:,:]
                _b=_b.expand(3,-1,-1)
                # save_rgb_image(_b,"../../Output/img_%03d_boundary_%01d.png"%(img_index,division))
                # save_image((pred_background*(1-mask))[k,:,:,:],"../../Output/img_%03d_background_pred.png"%img_index)
                os.makedirs(os.path.join("../../Data/Output_Results"),exist_ok=True)            
                img_index=batch_idx*args.batch_size+k
                save_image(x[k,:,:,:],os.path.join("../../Data/Output_Results",_meta_dir+"%03d.input.png"%(img_index)))
                save_image(img_background,os.path.join("../../Data/Output_Results",_meta_dir+"%03d.output.png"%(img_index)))
            if diff_masked<args.diff_threshold:
                break
        
            
                

end_time = time.time()
print("IEM finished in {:.1f} seconds".format(end_time-start_time))