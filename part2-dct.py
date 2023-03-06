import cv2
import numpy as np

from PIL import Image
from pathlib import Path
from tempfile import gettempdir

import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf


def _get_2d_dct_weight(d=8):
    cache_path = Path(gettempdir()) / f'dct{d}x{d}.pt'

    # check if the matrix is already saved
    if cache_path.is_file():
        dct_weight = torch.load(cache_path)
        return dct_weight

    # compute DCT matrix
    dct_mtx = np.zeros((d, d, d, d), dtype=np.float32)
    for hi in range(d):
        for wi in range(d):
            impulse = np.zeros((d,d), dtype=np.float32)
            impulse[hi,wi] = 1.0
            basis: np.ndarray = cv2.idct(impulse)
            dct_mtx[hi,wi,:,:] = basis

    # zig-zag
    dct_weight = []
    for i in range(2*d-1):
        if i < d:
            for j in range(0,i+1):
                if i % 2 == 0: # i = 0, 2, 4, 6, ...
                    basis = dct_mtx[i-j,j,:,:]
                else: # i = 1, 3, 5, ...
                    basis = dct_mtx[j,i-j,:,:]
                dct_weight.append(basis)
        else:
            for j in range(i+1-d, d):
                if i % 2 == 0: # i = 0, 2, 4, 6, ...
                    basis = dct_mtx[i-j,j,:,:]
                else: # i = 1, 3, 5, ...
                    basis = dct_mtx[j,i-j,:,:]
                dct_weight.append(basis)
    dct_weight = np.expand_dims(np.stack(dct_weight, axis=0), axis=1)
    assert dct_weight.shape == (d*d, 1, d, d)
    dct_weight = torch.from_numpy(dct_weight)

    # cache file
    assert not cache_path.is_file()
    if not cache_path.parent.is_dir():
        cache_path.parent.mkdir(parents=False)
    torch.save(dct_weight, cache_path)
    return dct_weight


# def get_dct(d=8, inverse=False):
#     if not inverse:
#         conv = nn.Conv2d(3, 3*d*d, kernel_size=d, stride=d, groups=3, bias=False)
#         assert conv.weight.shape == (3*d*d, 1, d, d)
#     else:
#         conv = nn.ConvTranspose2d(3*d*d, 3, kernel_size=d, stride=d, groups=3, bias=False)
#         assert conv.weight.shape == (3*d*d, 1, d, d)

#     conv.requires_grad_(False)
#     dct_weight = _get_2d_dct_weight(d=d)
#     assert dct_weight.shape == (d*d, 1, d, d)
#     conv.weight.data = dct_weight.repeat(3, 1, 1, 1)
#     return conv

def get_dct(d=8, inverse=False):
    if not inverse:
        conv = nn.Conv2d(1, d*d, kernel_size=d, stride=d, bias=False)
        assert conv.weight.shape == (d*d, 1, d, d)
    else:
        conv = nn.ConvTranspose2d(d*d, 1, kernel_size=d, stride=d, bias=False)
        assert conv.weight.shape == (d*d, 1, d, d)

    conv.requires_grad_(False)
    dct_weight = _get_2d_dct_weight(d=d)
    assert dct_weight.shape == (d*d, 1, d, d)
    conv.weight.data = dct_weight
    return conv


def imread_torch(impath):
    im = Image.open(impath).convert('L')
    im = tvf.to_tensor(im).unsqueeze_(0)
    return im


@torch.no_grad()
def main():
    # impath = Path('C:/Users/duanz/Downloads/sample_images/peppers.png')
    impath = Path('C:/Users/duanz/Downloads/sample_images/cat.png')
    d = 8

    # load image
    im = imread_torch(impath)
    H, W = im.shape[-2:]
    im = im[:, :, :H//d*d, :W//d*d]

    # initialize DCT
    dct2d = get_dct(d=d, inverse=False)
    idct2d = get_dct(d=d, inverse=True)

    # sanity check
    im_idct = idct2d(dct2d(im))
    assert torch.abs(im - im_idct).max() < 1e-5

    # DCT
    coeff = dct2d(im)

    for k in [2, 4, 8, 16, 32, 64]:
        # keep only the first k coefficients
        coeff_k = coeff.clone()
        coeff_k[:, k:, :, :] = 0
        im_hat = idct2d(coeff_k)
        assert im_hat.shape == im.shape
        mse = tnf.mse_loss(im, im_hat, reduction='mean')
        psnr = -10 * torch.log10(mse)
        print(f'{k=:3d}, {mse=:.4f}, {psnr=:.4f}')
        img_hat = tvf.to_pil_image(im_hat.squeeze_(0))
        img_hat.save(f'results/dct{k}.png')

    debug = 1


if __name__ == '__main__':
    main()
