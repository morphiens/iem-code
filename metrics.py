import torch

def neg_coeff_constraint(x, mask, pred_foreground, pred_background):
    # computes 2-C(F,B), which is equivalent to -C(F,B) as a loss term
    # we have 2-C(F,B) = H(F|B)/H(F) + H(B|F)/H(B), where:
    # H(F|B) = H(X*M | X*(1-M)) = ||M * (X - phi(X*(1-M), 1-M))||_1 = ||M * (X - pred_foreground)||_1
    # H(B|F) = H(X*(1-M) | X*M) = ||(1-M) * (X - phi(X*M, M))||_1 = ||(1-M) * (X - pred_background)||_1
    # H(F) = ||1-M||
    # H(B) = ||M||

    H_foreground_given_background = (mask*torch.abs(x-pred_foreground )).mean(1).sum((1,2))
    H_background_given_foreground = ((1-mask)*torch.abs(x-pred_background)).mean(1).sum((1,2))
    
    H_foreground = mask.sum((1,2,3))
    H_background = (1-mask).sum((1,2,3))

    C_foreground_given_background = - H_foreground_given_background / H_foreground
    C_background_given_foreground = - H_background_given_foreground / H_background
    C = C_foreground_given_background + C_background_given_foreground

    return -C

def switch_masks(x, mask, foreground, background,background_mask=None):
    #up_diffs
    # print(mask.nonzero().shape,(1-mask).nonzero().shape)
    eta=1e-16
    if background_mask is None:
        background_mask=1-mask
    
    mu_foreground = foreground.sum((2,3), keepdim=True) / (mask.sum((2,3), keepdim=True))
    mu_background = background.sum((2,3), keepdim=True) / (background_mask.sum((2,3), keepdim=True))

    mu_current=mu_foreground.expand((-1,-1,x.shape[-2],x.shape[-1]))*mask+mu_background.expand((-1,-1,x.shape[-2],x.shape[-1]))*background_mask
    mu_other=mu_foreground.expand(-1,-1,x.shape[-2],x.shape[-1])*background_mask+mu_background.expand(-1,-1,x.shape[-2],x.shape[-1])*mask
    sigma_current=(x-mu_current)
    sigma_other=(x-mu_other)
    sigma_current=torch.square(sigma_current).sum(dim=-3,keepdim=True)
    sigma_other=torch.square(sigma_other).sum(dim=-3,keepdim=True)
    update_bool= sigma_other<sigma_current
    
    return update_bool*((background_mask+mask)==1)


    
def diversity(x, mask, foreground, background):
    # computes sigma(M;X) + sigma(1-M;X), where:
    # sigma(M;X) = ||M * (X - mu(X*M))||_2^2
    # sigma(1-M;X) = ||(1-M) * (X - mu(X*(1-M)))||_2^2

    # here we use the sum over foreground pixels divided by the sum over mask values instead
    # of the mean over foreground pixels since we don't want to count masked-out pixels
    mu_foreground = foreground.sum((2,3), keepdim=True) / mask.sum((2,3), keepdim=True)
    mu_background = background.sum((2,3), keepdim=True) / (1-mask).sum((2,3), keepdim=True)

    sigma_foreground = (mask * (x - mu_foreground)**2).sum((1,2,3))
    sigma_background = ((1-mask) * (x - mu_background)**2).sum((1,2,3))

    return sigma_foreground + sigma_background
    
def diversity_updated(x, mask, foreground, background):
    # computes sigma(M;X) + sigma(1-M;X), where:
    # sigma(M;X) = ||M * (X - mu(X*M))||_2^2
    # sigma(1-M;X) = ||(1-M) * (X - mu(X*(1-M)))||_2^2

    # here we use the sum over foreground pixels divided by the sum over mask values instead
    # of the mean over foreground pixels since we don't want to count masked-out pixels
    mu_foreground = foreground.sum((2,3), keepdim=True) / (mask.sum((2,3), keepdim=True)+1e-16)
    mu_background = background.sum((2,3), keepdim=True) / ((1-mask).sum((2,3), keepdim=True)+1e-16)

    sigma_foreground = torch.sqrt((mask * (x - mu_foreground)**2).sum((1,2,3))/(mask.sum((1,2,3))+1e-16))
    sigma_background = torch.sqrt(((1-mask) * (x - mu_background)**2).sum((1,2,3))/(mask.sum((1,2,3))+1e-16))
    print(sigma_background,sigma_foreground)
    return sigma_foreground + sigma_background

def compute_performance(mask, mask2):
    # computes acc, IoU, mIoU and DICE score.
    # adapted from https://github.com/mickaelChen/ReDO

    acc = torch.max(
        ((mask >= .5).float() == mask2).float().mean(-1).mean(-1),
        ((mask <  .5).float() == mask2).float().mean(-1).mean(-1)).mean().item()

    iou = torch.max(
        ((((mask >= .5).float() + mask2.float()) == 2).float().sum(-1).sum(-1) /
         (((mask >= .5).float() + mask2.float()) >= 1).float().sum(-1).sum(-1)),
        ((((mask <  .5).float() + mask2.float()) == 2).float().sum(-1).sum(-1) /
         (((mask <  .5).float() + mask2.float()) >= 1).float().sum(-1).sum(-1))).mean().item()

    dice = torch.max(
        ((((mask >= .5).float() + mask2.float()) == 2).float().sum(-1).sum(-1)*2 /
         (((mask >= .5).float().sum(-1).sum(-1) + mask2.float().sum(-1).sum(-1))).float()),
        ((((mask <  .5).float() + mask2.float()) == 2).float().sum(-1).sum(-1)*2 /
         (((mask <  .5).float().sum(-1).sum(-1) + mask2.float().sum(-1).sum(-1))).float())).mean().item()

    inv_mask2 = 1 - mask2
    inv_iou = torch.max(
        ((((mask >= .5).float() + inv_mask2.float()) == 2).float().sum(-1).sum(-1) /
         (((mask >= .5).float() + inv_mask2.float()) >= 1).float().sum(-1).sum(-1)),
        ((((mask <  .5).float() + inv_mask2.float()) == 2).float().sum(-1).sum(-1) /
         (((mask <  .5).float() + inv_mask2.float()) >= 1).float().sum(-1).sum(-1))).mean().item()
    
    miou = (iou + inv_iou) / 2.0
    return acc, iou, miou, dice