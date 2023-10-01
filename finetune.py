import h5py
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision
import torch.optim as optim
from KDEF_dataset import dataset_train,dataset_test
from utils import load_model
# from torch.nn.parallel import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"]="7"


EPOCH = 30
BATCH_SIZE = 32

dir = "finetune2"



def finetune_model(teacher, train_loader,test_loader,cls,iter):
    teacher = teacher.cuda()
    # teacher = DataParallel(teacher, device_ids=[0, 1, 2, 3])
    teacher.train()
    accu_best = 0
    epoch_best = 0

    loss_func = torch.nn.CrossEntropyLoss()

    if cls == 'all':
        optimizer = optim.SGD(teacher.parameters(), lr=1e-3)
    elif cls == 'last':
        optimizer = optim.SGD(teacher.classifier.parameters(), lr=1e-3)

    for epoch in range(EPOCH):

        # teacher.train()
        # for i, (x, y) in enumerate(train_loader):
        #     x = x.type(torch.FloatTensor)
        #     y = y.long()
        #     b_x, b_y = x.cuda(), y.cuda()
        #     teacher_output = teacher(b_x)
        #     loss = loss_func(teacher_output, b_y)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5)
        #     optimizer.step()
        #
        #     if (i % 20 == 0):
        #         print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item())

        teacher.eval()
        num = 0
        total_num = 0

        for i, (x, y) in enumerate(test_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = teacher(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]

        accu1 = num / total_num

        print("Epoch:", epoch + 1, "accuracy:", accu1)

        # if accu1 > accu_best:
        #     torch.save(teacher.state_dict(), os.path.join(dir, "finetune" + str(iter) + ".pth"))
        #     accu_best = accu1
        #     epoch_best = epoch

    return accu_best,epoch_best



if __name__ == "__main__":
    test_data = dataset_test('dataset_test.h5')
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    train_data = dataset_train('dataset_attack.h5')
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

    if os.path.exists(dir) == 0:
        os.mkdir(dir)

    accus = []

    for iter in range(10):
        iters = iter

        if iters < 10:
            cls = 'all'
        elif 10 <= iters < 20:
            cls = 'last'

        # teacher = torchvision.models.vgg16_bn(pretrained=False)
        # in_feature = teacher.classifier[-1].in_features
        # teacher.classifier[-1] = torch.nn.Linear(in_feature, 7)
        # teacher.load_state_dict(torch.load("teacher/teacher.pth"))

        teacher = load_model(0,"teacher")



        teacher.eval()

        print("Beigin training model:", iters, "Model:", cls)
        accu,epoch_best = finetune_model(teacher,train_loader,test_loader,cls,iters)
        teacher = torchvision.models.vgg16_bn(pretrained=False)


        in_feature = teacher.classifier[-1].in_features
        teacher.classifier[-1] = torch.nn.Linear(in_feature, 7)
        teacher.load_state_dict(torch.load("teacher/teacher.pth"))
        teacher.eval()
        accus.append(accu)
        print("Model {} has been trained and the accuracy is {} with fine-tuning {} epoches".format(iters, accu,epoch_best))

    print(np.array(accus))