from numpy.core.numeric import False_
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms
import numpy as np
from torch import nn, optim
import cv2 as cv
import torch
from torchvision.models import resnet50
import argparse
from dataset import Face_Dataset
from tqdm import tqdm
from tensorboardX import SummaryWriter, writer
import h5py
from copy import deepcopy

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

train_dataset_loc = './data/KDEF_ORDER/'
test_dataset_loc = './data/KDEF_ORDER_TEST/'

train_dataset = Face_Dataset(train_dataset_loc, img_transforms)
# test_dataset = Face_Dataset(test_dataset_loc, img_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)

label_count = torch.zeros([7])
count_max = 300
imgs1 = []
imgs2 = []
imgs_test = []
labels1 = []
labels2 = []
labels_test = []


for i,(img,label) in enumerate(train_loader):
    img = img.squeeze().numpy()
    label = label.squeeze().numpy()
    label_count[label] += 1
    if label_count[label] <= count_max:
        imgs1.append(deepcopy(img))
        labels1.append(deepcopy(label))
    elif count_max < label_count[label] <= 2*count_max:
        imgs2.append(deepcopy(img))
        labels2.append(deepcopy(label))
    else:
        imgs_test.append(deepcopy(img))
        labels_test.append(deepcopy(label))


imgs1 = np.array(imgs1)
imgs2 = np.array(imgs2)
imgs_test = np.array(imgs_test)

labels1 = np.array(labels1)
labels2 = np.array(labels2)
labels_test = np.array(labels_test)

print(imgs1.shape,imgs2.shape,imgs_test.shape)
print(np.max(imgs1),np.min(imgs1))

file1 = h5py.File('data/dataset_defend.h5','w')
file1.create_dataset("/data",data=imgs1)
file1.create_dataset("/label",data=labels1)
file2 = h5py.File('data/dataset_attack.h5','w')
file2.create_dataset("/data",data=imgs2)
file2.create_dataset("/label",data=labels2)
file3 = h5py.File('data/dataset_test.h5','w')
file3.create_dataset("/data",data=imgs_test)
file3.create_dataset("/label",data=labels_test)


# print(label_count)
# print(torch.max(img),torch.min(img))
#
# img_total = torch.cat(img_total,dim=0)
# avg1,avg2,avg3 = torch.mean(img_total[:,0,:,:]),torch.mean(img_total[:,1,:,:]),torch.mean(img_total[:,2,:,:])
# std1,std2,std3 = torch.var(img_total[:,0,:,:]),torch.var(img_total[:,1,:,:]),torch.var(img_total[:,2,:,:])
#
# print(avg1,avg2,avg3)
# print(std1,std2,std3)























