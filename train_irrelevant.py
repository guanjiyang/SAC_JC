import numpy as np

from KDEF_dataset import dataset_test,dataset_train
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
import cv2
import time
# from torch.nn.parallel import DataParallel
os.environ['CUDA_VISIBLE_DEVICES']='0'


BATCH_SIZE = 64
EPOCH = 100

dir = "irrelevant"

def reset(cls):
    if cls == 'resnet':
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 7)
    elif cls == 'vgg':
        model = torchvision.models.vgg13_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 7)
    elif cls == 'dense':
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 7)

    elif cls == 'mobile':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 7)

    model.cuda()
    return model


def train_model(iter,cls):

    accu_best = 0
    test_data = dataset_test('dataset_test.h5')
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    train_data = dataset_train('dataset_attack.h5')
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    model = reset(cls)
    # model = DataParallel(model, device_ids=[0, 1, 2])

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    count = 0

    time1 = time.time()

    for epoch in range(EPOCH):


        model.train()
        for i, (x, y) in enumerate(train_loader):

            # if i == 0:
            #     img = x[0].numpy()
            #     img = 255*np.transpose(img,[1,2,0])
            #     cv2.imwrite("test.jpg",img)
            #     print("save")

            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item())

        if isnan(loss.data.item())==1:
            model = reset(cls)
            print("Reseting model!")

        scheduler.step()

        model.eval()
        num = 0
        total_num = 0

        for i, (x, y) in enumerate(test_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]

        accu1 = num / total_num

        time2 = time.time()

        print("Epoch:", epoch + 1, "accuracy:", accu1,"Time:", time2-time1)

        if accu1 > accu_best:
            torch.save(model.state_dict(), os.path.join(dir, "irrelevant" + str(iter) + ".pth"))
            accu_best = accu1



    return accu_best

if __name__ == "__main__":
    if os.path.exists(dir) == 0:
        os.mkdir(dir)

    accus = []

    for iter in range(20):
        iters = iter

        if iters < 5:
            cls = 'vgg'
        elif 5 <= iters < 10:
            cls = 'resnet'
        elif 10 <= iters < 15:
            cls = 'dense'
        elif 15 <= iters < 20:
            cls = 'mobile'

        print("Beigin training model:",iters,"clean model:",cls)
        accu = train_model(iters,cls)
        accus.append(accu)
        print("Model {} has been trained and the accuracy is {}".format(iters,accu))

    print(accus)

