import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class GuassianBlur(nn.Module):
    def __init__(self, sigma, kernel_size,reps):
        super().__init__()
        self.reps = reps
        self.padding = kernel_size//2
        squared_dists = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)**2
        gaussian_kernel = torch.exp(-0.5 * squared_dists / sigma**2)
        gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).view(1, 1, kernel_size).repeat(3,1,1)
        self.register_buffer('gaussian_kernel', gaussian_kernel)
        
    def gaussian_filter(self, x):
        # due to separability we can apply two 1d gaussian filters to get some speedup
        l=F.pad(x,(0,0,self.padding,self.padding),mode="replicate")
        v = F.conv2d(l, self.gaussian_kernel.unsqueeze(3), padding=0, groups=x.shape[-3])
        h = F.conv2d(F.pad(v,(self.padding,self.padding,0,0),mode="replicate"), self.gaussian_kernel.unsqueeze(2), padding=0, groups=x.shape[-3])
        return h
    def forward(self,x):
        u=x
        for _ in range(self.reps): u = self.gaussian_filter(u)
        return u

class Inpainter(nn.Module):
    def __init__(self, sigma, kernel_size, reps, scale_factor=1):
        super(Inpainter, self).__init__()
        self.reps = reps
        self.padding = kernel_size//2
        
        self.downsample = nn.AvgPool2d(scale_factor) if scale_factor > 1 else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest') if scale_factor > 1 else nn.Identity()
        squared_dists = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)**2
        gaussian_kernel = torch.exp(-0.5 * squared_dists / sigma**2)
        gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).view(1, 1, kernel_size).repeat(4,1,1)
        self.register_buffer('gaussian_kernel', gaussian_kernel)

    def gaussian_filter(self, x):
        # due to separability we can apply two 1d gaussian filters to get some speedup
        v = F.conv2d(x, self.gaussian_kernel.unsqueeze(3), padding=(self.padding,0), groups=4)
        h = F.conv2d(v, self.gaussian_kernel.unsqueeze(2), padding=(0,self.padding), groups=4)
        return h

    def forward(self, x, m):
        # to perform the same convolution on each channel of x and on the mask,
        # we concatenate x and m and perform a convolution with groups=num_channels=4
        u = torch.cat((x, m), 1)
        epsilon = u.sum((2,3), keepdim=True) * 1e-8
        u = self.downsample(u)
        for _ in range(self.reps): u = self.gaussian_filter(u)
        u = self.upsample(u)
        u = u + epsilon
        filtered_x = u[:,:-1]
        filtered_m = u[:,-1:]
        return filtered_x / filtered_m

class Boundary(nn.Module):
    def __init__(self):
        # a pixel belongs to the mask's boundary if its value is 1 and
        # it has at least one 0-valued neighboring pixel (non-diagonal).
        # we use a conv kernel that yields out[i,j] = 4*M[i,j] - (M[i+1,j] + M[i-1,j] + M[i,j+1] + M[i,j-1]),
        # hence a out[i,j] >= 0.5 if and only if the pixel (i,j) belongs to the boundary of M.
        super(Boundary, self).__init__()
        self.register_buffer('kernel', torch.Tensor([[0.,-1.,0.], [-1.,4.,-1.], [0.,-1.,0.]]).view((1,1,3,3)))
        
    def forward(self, m):
        fore_boundary = F.conv2d(m, self.kernel, padding=1) >= 0.5
        back_boundary = F.conv2d(1-m, self.kernel, padding=1) >= 0.5
        return fore_boundary + back_boundary
    

class BoundaryPixelCount(nn.Module):
    def __init__(self):
        # a pixel belongs to the mask's boundary if its value is 1 and
        # it has at least one 0-valued neighboring pixel (non-diagonal).
        # we use a conv kernel that yields out[i,j] = 4*M[i,j] - (M[i+1,j] + M[i-1,j] + M[i,j+1] + M[i,j-1]),
        # hence a out[i,j] >= 0.5 if and only if the pixel (i,j) belongs to the boundary of M.
        super().__init__()
        self.register_buffer('kernel', torch.Tensor([[0.,-1.,0.], [-1.,4.,-1.], [0.,-1.,0.]]).view((1,1,3,3)))
    def forward(self,m):
        self._reversed_padding_repeated_twice = (1, 1, 1, 1) 
        return torch.abs(F.conv2d(F.pad(m, self._reversed_padding_repeated_twice, mode="replicate"),self.kernel,padding=_pair(0)))
        # boundary=torch.abs(F.conv2d(m, self.kernel, padding=1,padding_mode="replicate"))
        # return boundary


class TransitionLoss(nn.Module):
    def __init__(self):
        # a pixel belongs to the mask's boundary if its value is 1 and
        # it has at least one 0-valued neighboring pixel (non-diagonal).
        # we use a conv kernel that yields out[i,j] = 4*M[i,j] - (M[i+1,j] + M[i-1,j] + M[i,j+1] + M[i,j-1]),
        # hence a out[i,j] >= 0.5 if and only if the pixel (i,j) belongs to the boundary of M.
        super().__init__()
        self.register_buffer('up_kernel', torch.Tensor([[0.,-1.,0.], [0.,1.,0], [0.,0.,0.]]).view((1,1,3,3)))
        self.register_buffer('down_kernel', torch.Tensor([[0.,0.,0.], [0.,1.,0], [0.,-1.,0.]]).view((1,1,3,3)))
        self.register_buffer('left_kernel', torch.Tensor([[0.,0.,0.], [-1.,1.,0], [0.,0.,0.]]).view((1,1,3,3)))
        self.register_buffer('right_kernel', torch.Tensor([[0.,0.,0.], [0.,1.,-1.], [0.,0.,0.]]).view((1,1,3,3)))
        
    def forward(self,x,mask):
        self._reversed_padding_repeated_twice = (1, 1, 1, 1) 
        losses=[]
        for k in ["up_kernel","down_kernel","left_kernel","right_kernel"]:
            if x.shape[-3]==1:
                kernel=getattr(self,k)
            else:
                kernel=getattr(self,k).expand(x.shape[-3],1,3,3)

            diffs=F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode="replicate"),kernel,padding=_pair(0),groups=x.shape[-3])
            diffs_mask=torch.abs(F.conv2d(F.pad(mask, self._reversed_padding_repeated_twice, mode="replicate"),getattr(self,k),padding=_pair(0)))            
            losses.append(torch.sum(torch.abs(diffs*diffs_mask),axis=1)/torch.sum(diffs_mask))
        
        losses=torch.stack(losses,axis=-1)
        loss=torch.sum(losses,axis=-1)
        return loss

class ContinuityLoss(nn.Module):
    def __init__(self):
        # a pixel belongs to the mask's boundary if its value is 1 and
        # it has at least one 0-valued neighboring pixel (non-diagonal).
        # we use a conv kernel that yields out[i,j] = 4*M[i,j] - (M[i+1,j] + M[i-1,j] + M[i,j+1] + M[i,j-1]),
        # hence a out[i,j] >= 0.5 if and only if the pixel (i,j) belongs to the boundary of M.
        super().__init__()
        self.register_buffer('up_kernel', torch.Tensor([[0.,-1.,0.], [0.,1.,0], [0.,0.,0.]]).view((1,1,3,3)))
        self.register_buffer('down_kernel', torch.Tensor([[0.,0.,0.], [0.,1.,0], [0.,-1.,0.]]).view((1,1,3,3)))
        self.register_buffer('left_kernel', torch.Tensor([[0.,0.,0.], [-1.,1.,0], [0.,0.,0.]]).view((1,1,3,3)))
        self.register_buffer('right_kernel', torch.Tensor([[0.,0.,0.], [0.,1.,-1.], [0.,0.,0.]]).view((1,1,3,3)))
        
    def forward(self,x,mask):
        self._reversed_padding_repeated_twice = (1, 1, 1, 1) 
        losses=[]
        for k in ["up_kernel"]:#,"down_kernel","left_kernel","right_kernel"]:
            if x.shape[-3]==1:
                kernel=getattr(self,k)
            else:
                kernel=getattr(self,k).expand(x.shape[-3],1,3,3)

            diffs=F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode="replicate"),kernel,padding=_pair(0),groups=x.shape[-3])
            diffs_mask=1-torch.abs(F.conv2d(F.pad(mask, self._reversed_padding_repeated_twice, mode="replicate"),getattr(self,k),padding=_pair(0)))            
            losses.append(torch.sum(torch.abs(diffs*diffs_mask),axis=1)/torch.sum(diffs_mask))
        
        losses=torch.stack(losses,axis=-1)
        loss=torch.sum(losses,axis=-1)
        return loss
