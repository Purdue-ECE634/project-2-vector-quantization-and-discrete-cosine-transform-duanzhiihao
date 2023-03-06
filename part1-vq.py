import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from scipy.spatial import distance
from sklearn.cluster import KMeans


def imread_gray_float32(impath):
    im = Image.open(impath).convert('L')
    im = np.array(im, dtype=np.float32) / 255.0
    return im


def run_vq(train_paths, test_path, codebook_size=128):
    block_size = 4
    # build training set
    im_train = [imread_gray_float32(impath) for impath in train_paths]
    X = [] # training set
    for im in im_train:
        # window partition
        H, W = im.shape
        nH, nW = H // block_size, W // block_size
        im = im[:nH * block_size, :nW * block_size]
        patches = im.reshape(H // block_size, block_size, W // block_size, block_size)
        patches = np.transpose(patches, axes=(0, 2, 1, 3)).reshape(-1, block_size, block_size)
        if False: # debug
            for p in patches:
                plt.imshow(p, cmap='gray')
                plt.show()
        # append to training set
        X.append(patches.reshape(-1, block_size * block_size))
    X = np.concatenate(X, axis=0)
    assert X.shape[1] == (block_size * block_size)

    # Run K-means (i.e., the Lloyd VQ algorithm)
    kmeans = KMeans(
        n_clusters=codebook_size, init='k-means++', n_init='auto', random_state=0
    ).fit(X)
    codebook = kmeans.cluster_centers_

    # test on a single image
    x = imread_gray_float32(test_path)
    H, W = x.shape
    nH, nW = H // block_size, W // block_size
    x_patches = x.reshape(nH, block_size, nW, block_size)
    x_patches = np.transpose(x_patches, axes=(0, 2, 1, 3)).reshape(-1, block_size*block_size)
    # nearest neighbor search
    dist_matrix = distance.cdist(x_patches, codebook, metric='euclidean')
    code_idx = np.argmin(dist_matrix, axis=1)
    # reconstruction
    rec_patches = codebook[code_idx]
    assert rec_patches.shape == (nH * nW, block_size * block_size)
    rec = rec_patches.reshape(nH, nW, block_size, block_size).transpose(0, 2, 1, 3).reshape(H, W)
    if False:
        plt.imshow(rec, cmap='gray')
        plt.show()

    # compute metrics
    mse = np.mean(np.square(rec - x))
    psnr = -10 * np.log10(mse)
    print(f'{len(train_paths)=}, {test_path.name=}, {codebook_size=}, {mse=:.4f}, {psnr=:.4f}')

    if True: # save images
        plt.imsave(f'./results/vq_original.png', x, cmap='gray')
        plt.imsave(f'./results/vq_rec_train{len(train_paths)}_L{codebook_size}.png', rec, cmap='gray')


def main():
    root = Path('C:/Users/duanz/Downloads/sample_images')
    image_paths = sorted(root.rglob('*.png'))
    # run_vq(image_paths[:1], image_paths[0], codebook_size=128)
    # run_vq(image_paths[:1], image_paths[0], codebook_size=256)
    # run_vq(image_paths[1:11], image_paths[0], codebook_size=128)
    # run_vq(image_paths[1:11], image_paths[0], codebook_size=256)
    run_vq(image_paths[-2:-1], image_paths[-2], codebook_size=128)
    run_vq(image_paths[-2:-1], image_paths[-2], codebook_size=256)
    run_vq(image_paths[1:11], image_paths[-2], codebook_size=128)
    run_vq(image_paths[1:11], image_paths[-2], codebook_size=256)


if __name__ == '__main__':
    main()
