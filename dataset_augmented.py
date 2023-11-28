from KDEF_dataset import dataset_test,dataset_none
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
import numpy as np
import h5py
from imagecorruptions import corrupt
from copy import deepcopy
import time

BATCH_SIZE = 1
transform_test = transforms.Compose([transforms.ToTensor()])
test_data = dataset_none('dataset_test.h5')
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

def augment_data(iter,augment_name):
    imgs = []
    labels = []
    batch_size = 800

    Sever = int(np.floor(iter/2))+1

    if iter % 2 == 0:
        flip_flag = 0
    else:
        flip_flag = 1

    print("Severity:",Sever,"flip:",flip_flag)


    for i, (input, target) in enumerate(test_loader):


        if  i < batch_size:

            image = deepcopy(input)
            image = torch.squeeze(image, dim=0).permute(1, 2, 0)
            image = torch.floor(255 * image).numpy().astype(np.uint8)
            image = corrupt(image, corruption_name=augment_name, severity=Sever)
            image = transform_test(image).unsqueeze(dim=0)
            imgs.append(image)
            labels.append(target)


    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    print(imgs.shape, labels.shape)

    if flip_flag == 1:
        imgs = imgs.transpose(2,3)


    print(imgs.shape, torch.max(imgs), torch.min(imgs))

    imgs = np.array(imgs)
    labels = np.array(labels)


    root = 'data/data_augmented/'+augment_name+'_augment_s={}_f={}.h5'.format(Sever,flip_flag)


    file1 = h5py.File(root, 'w')
    file1.create_dataset("/data", data=imgs)
    file1.create_dataset("/label", data=labels)

if __name__ == '__main__':

    dir = 'data/data_augmented'
    if os.path.exists(dir) == 0:
        os.mkdir(dir)

    # for augment_name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
    #                  'zoom_blur', 'snow', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
    #                  'jpeg_compression']:

    for augment_name in ['jpeg_compression']:

        print(augment_name)
        for iter in range(2):

            augment_data(iter,augment_name)

