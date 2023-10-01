import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import time
import h5py
from KDEF_dataset import dataset_train,dataset_test,dataset_none
from collections import OrderedDict



class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.output = output

    def close(self):
        self.hook.remove()

def load_model(num, mode):
    if mode == 'teacher':
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 7)
        model.load_state_dict(torch.load("teacher/teacher.pth"))

    elif mode == 'irrelevant':
        if num<5:
            model = torchvision.models.vgg13_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        elif 5 <= num < 10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 7)

        elif 15>num>=10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 7)

        elif 20>num>=15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        model.load_state_dict(torch.load(os.path.join("irrelevant", "irrelevant" + str(num) + ".pth")))

    elif mode == 'model_extract':
        if num < 5:
            model = torchvision.models.vgg13_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        elif 5 <= num < 10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 7)

        elif 15 > num >= 10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 7)

        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        model.load_state_dict(torch.load(os.path.join("model_extract", "model_extract" + str(num) + ".pth")))

    elif mode == 'model_extract_kd':
        if num < 5:
            model = torchvision.models.vgg13_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        elif 5 <= num < 10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 7)

        elif 15 > num >= 10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 7)

        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        model.load_state_dict(torch.load(os.path.join("model_extract", "model_extract_kd" + str(num) + ".pth")))

    elif mode == 'adv_train':
        if num < 5:
            model = torchvision.models.vgg13_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        elif 5 <= num < 10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 7)

        elif 15 > num >= 10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 7)

        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        state_dict = torch.load(os.path.join("adv_train", "adv_train" + str(num) + ".pth"))

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
            new_state_dict[name] = v  # 新字典的key值对应的value一一对应

        model.load_state_dict(new_state_dict)


    elif mode == 'finetune':

        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        if num >= 10:
            model.load_state_dict(torch.load('finetune/finetune'+str(num)+'.pth'))

        elif num <10:
            model.load_state_dict(torch.load('finetune2/finetune' + str(num) + '.pth'))

    elif mode == "fine-pruning":
        model = torch.load("prune2/prune_model_" + str(num)+".pth")


    elif mode == 'CEM_extract':
        if num < 5:
            model = torchvision.models.vgg13_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        elif 5 <= num < 10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 7)

        elif 15 > num >= 10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 7)

        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        model.load_state_dict(torch.load(os.path.join("CEM", "CEM_extract" + str(num) + ".pth")))

    elif mode == 'CEM_irrelevant':
        if num < 5:
            model = torchvision.models.vgg13_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        elif 5 <= num < 10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 7)

        elif 15 > num >= 10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 7)

        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 7)

        model.load_state_dict(torch.load(os.path.join("CEM", "CEM_irrelevant" + str(num) + ".pth")))


    return model
