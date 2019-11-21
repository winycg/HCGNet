import sys
sys.path.append("..")
import foolbox
import torch
import torchvision.models as models
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import models

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# get source image and label
val_dir = '/data/Imagenet/val/'
val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False,
                                         num_workers=16, pin_memory=(torch.cuda.is_available()))


def adversarial_num(output):
    correct_num = 0
    for i in range(output.size(0)):
        if torch.isnan(output[i])[0, 0, 0]:
            correct_num += 1
    return correct_num


def acc_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    number = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        number.append(correct_k)
    return number




def attack_model(arch, epsilon):
    if arch.startswith('HCGNet'):
        net = getattr(models, arch)
        model = net(num_classes=1000).cuda()
        model = torch.nn.DataParallel(model).eval()
        checkpoint = torch.load('../checkpoint/HCGNet_B_best.pth.tar')
        model.load_state_dict(checkpoint['net'])
    else:
        model = torchvision.models.__dict__[arch](pretrained=True)
        model = torch.nn.DataParallel(model).eval()

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(
        model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

    # apply attack on source image
    attack = foolbox.attacks.FGSM(fmodel)

    total = 0
    ori_top1_correct = 0
    adv_top1_correct = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs, targets
        '''
        ori_output = fmodel.forward(inputs.numpy())
        ori_top1_correct += acc_num(torch.from_numpy(ori_output), targets, (1,))[0]
        '''
        adversarial_input = attack(inputs.numpy(), targets.numpy(), epsilons=[epsilon], max_epsilon=0)
        #adv_outputs = fmodel.forward(adversarial_input)

        adv_top1_correct += adversarial_num(torch.from_numpy(adversarial_input))
        print(batch_idx, adv_top1_correct)
        total += targets.size(0)
    #ori_top1_acc = ori_top1_correct / total
    adv_top1_acc = adv_top1_correct / total

    #print('original top1 accuracy:', ori_top1_acc)
    print('adversarial top1 accuracy:', adv_top1_acc)
    return adv_top1_acc


for arch in ['resnet18', 'densenet169', 'resnext50_32x4d', 'wide_resnet50_2', 'HCGNet_B']:
    for epsilon in [0.001, 0.002, 0.003, 0.004, 0.005]:
        x = attack_model(arch, epsilon)
        with open('attack_result.txt', 'a+') as f:
            f.write(arch + ' ' + str(x) + '\n')

