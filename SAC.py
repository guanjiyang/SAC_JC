import numpy as np

from KDEF_dataset import dataset_SAC,dataset_test
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from utils import load_model,FeatureHook
from sklearn.metrics import roc_curve,auc

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

BATCH_SIZE = 16





def correlation(m,n):
    m = F.normalize(m,dim=-1)
    n = F.normalize(n,dim=-1).transpose(0,1)
    cose = torch.mm(m,n)
    matrix = 1-cose
    return matrix



def pairwise_euclid_distance(A):
    sqr_norm_A = torch.unsqueeze(torch.sum(torch.pow(A, 2),dim=1),dim=0)
    sqr_norm_B = torch.unsqueeze(torch.sum(torch.pow(A, 2), dim=1), dim=1)
    inner_prod = torch.matmul(A, A.transpose(0,1))
    tile1 = torch.reshape(sqr_norm_A,[A.shape[0],1])
    tile2 = torch.reshape(sqr_norm_B,[1,A.shape[0]])
    return tile1+tile2 - 2*inner_prod


def correlation_dist(A):
    A = F.normalize(A,dim=-1)
    cor = pairwise_euclid_distance(A)
    cor = torch.exp(-cor)

    return cor



def cal_cor(model,dataloader):
    model.eval()
    model = model.cuda()
    outputs = []
    for i,(x,y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        #output = output.cpu().detach()
        outputs.append(output.detach())

    output = torch.concat(outputs,0)
    cor_mat = correlation(output,output)
    # cor_mat = correlation_dist(output)



    model = model.cpu()
    return cor_mat


def cal_correct(model,dataloader):
    model.eval()
    model = model.cuda()
    num = 0
    total_num = 0

    for i, (x, y) in enumerate(dataloader):
        x = x.type(torch.FloatTensor)
        # print(torch.max(x),torch.min(x))
        y = y.long()
        b_x, b_y = x.cuda(), y.cuda()
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()
        num += (pred == b_y).sum().item()
        total_num += pred.shape[0]

    acc = num/total_num

    model = model.cpu()
    return acc

def feature(model,dataloader):
    model.eval()
    model = model.cuda()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
    output = output.cpu().detach()
    model = model.cpu()
    return output

def similarity(output1,output2):
    sim = 0

    m = F.normalize(output1, dim=-1)
    n = F.normalize(output2, dim=-1).transpose(0, 1)
    for i in range(m.shape[0]):
        a = m[i,:].unsqueeze(0)
        b = n[:,i].unsqueeze(1)
        cose = torch.mm(a, b)
        sine = 1-cose
        sim += torch.abs(sine)

    return sim
def calculate_auc(list_a, list_b):
    l1,l2 = len(list_a),len(list_b)
    y_true,y_score = [],[]
    for i in range(l1):
        y_true.append(0)
    for i in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def cal_corre(models,iter,augment_name):
    cor_mats = []

    Sever = int(np.floor(iter / 2)) + 1

    # Sever = iter + 1

    if iter % 2 == 0:
        flip_flag = 0
    else:
        flip_flag = 1

    # flip_flag = 0

    data_root = 'data_augmented/'+augment_name+'_augment_s={}_f={}.h5'.format(Sever,flip_flag)

    # data_root ='data_augmented/normal_test.h5'
    # train_data = dataset_SAC(data_root)
    train_data = dataset_test(data_root)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    for i in range(len(models)):
        model = models[i]
        cor_mat = cal_cor(model, train_loader)
        cor_mats.append(cor_mat)


    diff = torch.zeros(len(models))

    for i in range(len(models) - 1):
        diff[i] = torch.sum(torch.abs(cor_mats[i + 1] - cor_mats[0]))/(cor_mat.shape[0]*cor_mat.shape[1])

    # print("Correlation difference is:", diff)
    # print("Correlation difference is:", diff[:20])
    # print("Correlation difference is:", diff[20:40])
    # print("Correlation difference is:", diff[40:60])
    # print("Correlation difference is:", diff[60:80])
    # print("Correlation difference is:", diff[80:100])
    # print("Correlation difference is:", diff[100:110])

    list1 = diff[:20]
    list2 = diff[20:40]
    list3 = diff[40:60]
    list4 = diff[60:80]
    list5 = diff[80:90]
    list6 = diff[90:100]
    list7 = diff[100:105]

    auc_p = calculate_auc(list1, list3)
    auc_l = calculate_auc(list2, list3)
    auc_adv = calculate_auc(list4, list3)
    auc_finetune_all = calculate_auc(list5, list3)
    auc_finetune_last = calculate_auc(list6, list3)
    auc_prune = calculate_auc(list7, list3)


    auc_list = [auc_p,auc_l,auc_adv,auc_finetune_all,auc_finetune_last,auc_prune]



    return auc_list



if __name__ == '__main__':

    models = []

    for i in [0]:
        globals()['teacher' + str(i)] = load_model(i, "teacher")
        models.append(globals()['teacher' + str(i)])

    for i in range(20):
        globals()['student_kd' + str(i)] = load_model(i, "model_extract_kd")
        models.append(globals()['student_kd' + str(i)])

    for i in range(20):
        globals()['student' + str(i)] = load_model(i , "model_extract")
        models.append(globals()['student' + str(i)])

    for i in range(20):
        globals()['clean' + str(i)] = load_model(i, "irrelevant")
        models.append(globals()['clean' + str(i)])

    for i in range(20):
        globals()['adv_train' + str(i)] = load_model(i, "adv_train")
        models.append(globals()['adv_train' + str(i)])

    for i in range(20):
        globals()['fine-tune' + str(i)] = load_model(i, "finetune")
        models.append(globals()['fine-tune' + str(i)])


    for i in range(5):
        globals()['fine-pruning' + str(i)] = load_model(i, "fine-pruning")
        models.append(globals()['fine-pruning' + str(i)])

    auc_total = []
    auc_avg_best_total = []

    # for augment_name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
    #                      'zoom_blur', 'snow', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
    #                      'jpeg_compression']:

    # for augment_name in ['jpeg_compression']:
    #
    #     for i in range(5):
    #
    #         for soft_factor in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5]:
    #
    #             auc_avg_list = []
    #             iter = i
    #             print("Augment:",augment_name,"Iter:", iter,"Soft_factor:",soft_factor)
    #             # auc_list = cal_corre(models,iter,augment_name)
    #             auc_list = cal_corre_soft(models, iter, augment_name,soft_factor)
    #
    #             auc_list = np.array(auc_list)
    #             # print("AUC for:",augment_name,iter)
    #             print("AUC_P:", auc_list[0], "AUC_L:", auc_list[1], "AUC_Adv:",  auc_list[2],"AUC_Finetune_all:", auc_list[3],"AUC_Finetune_last:", auc_list[4], "AUC_Prune:", auc_list[5])
    #
    #         # auc_avg = np.mean(auc_list)
    #         # auc_total.append(auc_list)
    #         # auc_avg_list.append(auc_avg)
    #
    #
    #     # auc_avg_best = np.max(np.array(auc_avg_list))
    #     # print("Augment:",augment_name,"AUC_best:",auc_avg_best)
    #     # auc_avg_best_total.append(auc_avg_best)
    #
    # # auc_total = np.array(auc_total)
    # # auc_avg_best_total = np.array(auc_avg_best_total)
    # # print(auc_avg_best_total)
    #
    #
    #
    # # np.save("data/auc_total.npy",auc_total)
    # # np.save("data/auc_avg_best_total.npy", auc_avg_best_total)


    for augment_name in ['jpeg_compression']:

        for i in range(1):

            auc_avg_list = []
            iter = i
            print("Augment:",augment_name,"Iter:", iter)
            # auc_list = cal_corre(models,iter,augment_name)
            auc_list = cal_corre(models, iter, augment_name)

            auc_list = np.array(auc_list)
            # print("AUC for:",augment_name,iter)
            print("AUC_P:", auc_list[0], "AUC_L:", auc_list[1], "AUC_Adv:",  auc_list[2],"AUC_Finetune_all:", auc_list[3],"AUC_Finetune_last:", auc_list[4], "AUC_Prune:", auc_list[5])

            # auc_avg = np.mean(auc_list)
            # auc_total.append(auc_list)
            # auc_avg_list.append(auc_avg)


        # auc_avg_best = np.max(np.array(auc_avg_list))
        # print("Augment:",augment_name,"AUC_best:",auc_avg_best)
        # auc_avg_best_total.append(auc_avg_best)

    # auc_total = np.array(auc_total)
    # auc_avg_best_total = np.array(auc_avg_best_total)
    # print(auc_avg_best_total)



    # np.save("data/auc_total.npy",auc_total)
    # np.save("data/auc_avg_best_total.npy", auc_avg_best_total)