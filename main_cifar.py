import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import shutil

import os
import argparse
import numpy as np

import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds

from bisect import bisect_right
import time
import math
from regularization.dropblock import LinearScheduler, SGDRScheduler


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='data', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
parser.add_argument('--arch', default='HCGNet_A1', type=str, help='network architecture')
parser.add_argument('--lr-type', default='SGDR', type=str, help='learning rate strategy')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')

parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--milestones', default=[150, 225], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=10, type=int, dest='sgdr_t',help='SGDR T_0')
parser.add_argument('--epochs', type=int, default=1270, help='number of epochs to train')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if not os.path.isdir('result'):
    os.mkdir('result')
if args.resume is False:
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + args.arch + '.txt', 'a+') as f:
        f.seek(0)
        f.truncate()


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
drop_scheduler = SGDRScheduler  # or 'LinearScheduler'
if drop_scheduler is LinearScheduler:
    drop_scheduler.num_epochs =args.epochs
# -----------------------------------------------------------------------------------------
# dataset
if args.dataset == 'cifar10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,
                                            transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                  [0.24703233, 0.24348505, 0.26158768])
                                            ]))
    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                           transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                 [0.24703233, 0.24348505, 0.26158768]),]))

elif args.dataset == 'cifar100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                             transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                     [0.2675, 0.2565, 0.2761])
                                             ]))

    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                     [0.2675, 0.2565, 0.2761]),
                                            ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes)
print('Params: %.2fM, Multi-adds: %.2fG'
      % (cal_param_size(net)/1e6, cal_multi_adds(net,)/1e9))
del(net)

net = model(num_classes=num_classes).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if args.evaluate:
        checkpoint = torch.load('./checkpoint/' + model.__name__ + '_best.pth.tar')
    else:
        checkpoint = torch.load('./checkpoint/' + model.__name__ + '.pth.tar')
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1


def adjust_lr(optimizer, epoch, eta_max=0.1, eta_min=0.):
    cur_lr = 0.
    if args.lr_type == 'SGDR':
        i = int(math.log2(epoch / args.sgdr_t + 1))
        T_cur = epoch - args.sgdr_t * (2 ** (i) - 1)
        T_i = (args.sgdr_t * 2 ** i)

        cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    elif args.lr_type == 'multistep':
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if args.arch == 'HCGNet_A2' or args.arch == 'HCGNet_A3' and epoch > 629:
        epoch = epoch - 630

    lr = adjust_lr(optimizer, epoch)
    drop_scheduler.global_epoch = epoch
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        iter_start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 1.)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        print('Epoch:{}\t batch_idx:{}/All_batch:{}\t lr:{:.3f}\t duration:{:.3f}\ttrain_acc:{:.2f}'
          .format(epoch, batch_idx, len(trainloader), lr, time.time()-iter_start_time, 100. * correct/total))
        iter_start_time = time.time()
    print('Epoch:{0}\t lr:{1:.3f}\t duration:{2:.3f}\ttrain_acc:{3:.2f}\ttrain_loss:{4:.6f}'
          .format(epoch, lr, time.time()-start_time, 100. * correct/total, train_loss/len(trainset)))
    
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + args.arch + '.txt', 'a+') as f:
        f.write('Epoch:{0}\t lr:{1:.3f}\t duration:{2:.3f}\ttrain_acc:{3:.2f}\ttrain_loss:{4:.6f}'
          .format(epoch, lr, time.time()-start_time, 100. * correct/total, train_loss/len(trainset)))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Epoch:{}\t batch_idx:{}/All_batch:{}\ttest_acc:{:.2f}'
            .format(epoch, batch_idx, len(testloader), 100. * correct/total))

        with open('result/' + str(os.path.basename(__file__).split('.')[0]) + args.arch + '.txt', 'a+') as f:
            f.write('\t test_acc:{0:.2f}\t test_loss:{1:.6f}'
                    .format(test_loss / len(testset), 100. * correct / total)+'\n')

    # Save checkpoint.
    acc = 100. * correct/total
    print('Test accuracy: ', acc)
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+model.__name__+'.pth.tar')

    is_best = False
    if best_acc < acc:
        best_acc = acc
        is_best = True

    if is_best:
        shutil.copyfile('./checkpoint/' + str(model.__name__) + '.pth.tar',
                        './checkpoint/' + str(model.__name__) + '_best.pth.tar')
    print('Save Successfully')
    print('------------------------------------------------------------------------')


if __name__ == '__main__':
    if args.evaluate:
        test(start_epoch)
    else:
        for epoch in range(start_epoch, args.epochs):
            train(epoch)
            test(epoch)






