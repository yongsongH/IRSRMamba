import lpips
import torch
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, net ='vgg',use_gpu=False, **kwargs):
    """Calculate LPIPS.

    Ref: https://github.com/richzhang/PerceptualSimilarity

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        net(str): net used for inference, it can be 'vgg' or 'alex'
        (loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
        loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization)

    Returns:
        float: lpips result.
    """

    loss_fn = lpips.LPIPS(net = 'vgg') # best forward scores
    img = lpips.im2tensor(img).detach()  # RGB image from [-1,1]
    img2 = lpips.im2tensor(img2).detach()
    print()
    if use_gpu:
        loss_fn.cuda()
        img = img.cuda()
        img2 = img2.cuda()

    d = loss_fn(img, img2).detach().item()

    return np.array(d)