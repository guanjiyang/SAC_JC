from KDEF_dataset import dataset_train,dataset_test
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from utils import load_model
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
import h5py
import time
from torch.nn.parallel import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,5,6"
dir = "adv_train"
BATCH_SIZE = 128


def PGD(model,image,label):

    model_surr = deepcopy(model)
    model_surr = model_surr.cuda()
    label = label.cuda()
    loss_func1 = torch.nn.CrossEntropyLoss()
    image_attack = deepcopy(image)
    image_attack = Variable(image_attack, requires_grad=True)
    image_attack = image_attack.cuda()
    alpha = 1/255
    epsilon = 4/255

    for iter in range(10):

        image_attack = image_attack.cuda()
        image_attack = Variable(image_attack, requires_grad=True)

        output = model_surr(image_attack)
        loss = -loss_func1(output,label)
        loss.backward()
        grad = image_attack.grad.detach().sign()
        image_attack = image_attack.detach()
        image_attack -= alpha*grad
        eta = torch.clamp(image_attack-image,min=-epsilon,max=epsilon)
        image_attack = torch.clamp(image+eta,min=0,max=1)


    pred_prob = output.detach()
    pred = torch.argmax(pred_prob, dim=-1)
    acc_num = torch.sum(label==pred)
    num = label.shape[0]
    acc = acc_num/num
    acc = acc.data.item()

    return image_attack.detach(),acc


def adv_train(model,teacher,train_loader,test_loader,iter):
    model = model.cuda()
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    model.train()
    teacher.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = 5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    time1 = time.time()

    for epoch in range(10):

        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
            x_adv,acc = PGD(model,b_x,b_y)
            output = model(b_x)
            x_adv = x_adv.cuda()
            output_adv = model(x_adv)
            loss_norm = loss_func(output, pred)
            loss_adv = loss_func(output_adv, pred)
            loss1 = loss_norm + loss_adv/(loss_adv.detach())
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

            if (i % 5 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss1.data.item(),loss_norm.data.item(),loss_adv.data.item(),"ASR:",1-acc)

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

        print("Epoch:", epoch + 1, "accuracy:", accu1,"Time:",time2-time1)

    torch.save(model.state_dict(), os.path.join(dir, "adv_train" + str(iter) + ".pth"))
    model = model.cpu()

    return accu1


if __name__ == "__main__":

    if os.path.exists(dir)==0:
        os.mkdir(dir)

    teacher = load_model(0, "teacher")
    teacher = teacher.cuda()
    teacher = DataParallel(teacher, device_ids=[0, 1, 2, 3])
    teacher.eval()

    models = []
    accus = []
    for i in range(20):
        iters = i
        globals()['student' + str(iters)] = load_model(iters, "model_extract")
        models.append(globals()['student' + str(iters)])

    test_data = dataset_test('dataset_test.h5')
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    train_data = dataset_train('dataset_attack.h5')
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

    for i in range(len(models)):
        accu = adv_train(models[i], teacher, train_loader, test_loader, iter=3)
        print("Adversarial training model:",str(i))
        accus.append(accu)
