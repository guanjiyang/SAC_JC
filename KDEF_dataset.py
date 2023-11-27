import h5py
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torchvision import transforms
import cv2
import os
from PIL import Image

def normalize(image):
    image_data = [[0.2342, 0.3056, 0.4314], [0.0232, 0.0381, 0.0525]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape)
    for i in range(3):
        img_copy[ i, :, :] = (image[ i, :, :] - image_data[0, i])/image_data[1,i]

    return img_copy


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(256, padding=32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ])

transform_none = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ])

class dataset_train(Dataset):
    def __init__(self, name):
        super(dataset_train, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data",name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = torch.tensor(image)
        image = transform_train(image)


        return [image,label]


class dataset_test(Dataset):
    def __init__(self, name):
        super(dataset_test, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data",name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        # image = torch.tensor(image)
        image = transform_test(image)


        return [image,label]


class dataset_none(Dataset):
    def __init__(self, name):
        super(dataset_none, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data",name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = torch.tensor(image)
        image = transform_none(image)

        return [image,label]


class dataset_SAC(Dataset):
    def __init__(self, name):
        super(dataset_SAC, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data",name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = np.transpose(image,[2,0,1])
        image = torch.tensor(image)
        image = transform_test(image)


        return [image,label]

class dataset_SAC_num(Dataset):
    def __init__(self, name,num):
        super(dataset_SAC_num, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data",name), 'r')
        self.images = np.array(self.data['/data'])[:num]
        self.labels = np.array(self.data['/label'])[:num]

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = np.transpose(image,[2,0,1])
        image = torch.tensor(image)
        image = transform_test(image)


        return [image,label]

