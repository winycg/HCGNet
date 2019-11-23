'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import shutil
import argparse
import numpy as np

import models

from utils import cal_param_size, cal_multi_adds
from bisect import bisect_right
import time
import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--num-classes', default=1000, type=int, help='classes number')
parser.add_argument('--arch', default='HCGNet_B', type=str, help='network architecture')
parser.add_argument('--train_data', default='/data/ImageNet/train/', type=str, help='train data location')
parser.add_argument('--val_data', default='/data/ImageNet/val/', type=str, help='validation data location')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size per process')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-type', default='SGDR', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[30, 60], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=10, type=int, dest='sgdr_t',help='SGDR T_0')
parser.add_argument('--epochs', type=int, default=630, help='number of epochs to train')
parser.add_argument('--gpu-id', type=str, default='0,1')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str,  default='./checkpoint/HCGNet_B_best.pth.tar', help='checkpoint file')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if args.resume is False:
    with open('result/'+ os.path.basename(__file__).split('.')[0] +'.txt', 'a+') as f:
        f.seek(0)
        f.truncate()

cudnn.benchmark = True
# cudnn.deterministic = True
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Model
model = getattr(models, args.arch)
net = model(num_classes=args.num_classes)
if args.local_rank == 0:
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net)/1e6, cal_multi_adds(net, (2, 3, 224, 224))/1e6))
del(net)


is_distributed = False
if 'WORLD_SIZE' in os.environ:
    is_distributed = int(os.environ['WORLD_SIZE']) > 1

args.gpu = 0
args.world_size = 1

if is_distributed:
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

net = model(num_classes=args.num_classes)

if args.sync_bn:
    import apex
    print("using apex synced BN")
    net = apex.parallel.convert_syncbn_model(net)

net = net.cuda()
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

train_dataset = datasets.ImageFolder(
    args.train_data,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(), Too slow
        # normalize,
    ]))
val_dataset = datasets.ImageFolder(args.val_data, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
]))


train_sampler = None
val_sampler = None
if is_distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

testloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True,
    sampler=val_sampler,
    collate_fn=fast_collate)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


criterion = nn.CrossEntropyLoss()
def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=1000, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1-epsilon)
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5, nesterov=True)


def mixup_data(x, y, alpha=0.4, use_cuda=True):
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
    return lam * criterion(pred, y_a, args.num_classes) + (1 - lam) * criterion(pred, y_b, args.num_classes)


if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.evaluate:
        resume_path = args.checkpoint
    else:
        resume_path = './checkpoint/' + model.__name__ + '.pth.tar'
    checkpoint = torch.load(resume_path)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']


def adjust_lr(optimizer, epoch, eta_max=args.init_lr, eta_min=0.):
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

def train(epoch):
    net.train()
    train_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    lr = adjust_lr(optimizer, epoch, args.init_lr * float(args.batch_size * args.world_size) / 256)
    start_time = time.time()
    batch_time = time.time()

    prefetcher = data_prefetcher(trainloader)
    inputs, targets = prefetcher.next()

    i = 0
    while inputs is not None:
        i += 1
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 0.4)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = mixup_criterion(CrossEntropyLoss_label_smooth, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num = acc_num(outputs, targets, (1, 5))
        top1_correct += num[0]
        top5_correct += num[1]
        total += targets.size(0)
        if args.local_rank == 0:
            print('Epoch:{0}\tBatch:{1}\t lr:{2:.6f}\t duration:{3:.3f}\ttop1_acc:{4:.2f}\ttop5_acc:{5:.2f}'
                  .format(epoch, i, lr, time.time() - batch_time, num[0]/targets.size(0), num[1]/targets.size(0)))
        batch_time = time.time()

        inputs, targets = prefetcher.next()

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    torch.cuda.empty_cache()
    if args.local_rank == 0:
        print('Epoch:{0}\t lr:{1:.6f}\t duration:{2:.3f}\ttop1_acc:{3:.2f}\ttop5_acc:{4:.2f}\ttrain_loss:{5:.6f}'
              .format(epoch, lr, time.time() - start_time, top1_acc, top5_acc,
                      train_loss / total))
        with open('result/' + os.path.basename(__file__).split('.')[0] + '.txt', 'a+') as f:
            f.write(str(time.time()-start_time)+' '+str(top1_acc) + ' ' + str(top5_acc) + ' ' + str(train_loss / total))

def test(epoch):
    net.eval()
    global best_acc
    test_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0

    prefetcher = data_prefetcher(testloader)
    inputs, targets = prefetcher.next()

    with torch.no_grad():
        i = 0
        while inputs is not None:
            i += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            num = acc_num(outputs, targets, (1, 5))
            top1_correct += num[0]
            top5_correct += num[1]
            total += targets.size(0)

            inputs, targets = prefetcher.next()

        top1_acc = top1_correct / total
        top5_acc = top5_correct / total
        if args.local_rank == 0:
            with open('result/'+ os.path.basename(__file__).split('.')[0] +'.txt', 'a+') as f:
                f.write(','+str(top1_acc)+' '+str(top5_acc)+' '+str(test_loss/total)+'\n')
    torch.cuda.empty_cache()

    if args.local_rank == 0:
        print('Test top1 accuracy: ', top1_acc)
        print('Test top5 accuracy: ', top5_acc)
        print('Test loss: ', test_loss/total)

        is_best = False
        if best_acc < top1_acc:
            best_acc = top1_acc
            is_best = True

        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+model.__name__+'.pth.tar')

        if is_best:
            shutil.copyfile('./checkpoint/'+model.__name__+'.pth.tar', './checkpoint/'+model.__name__+'_best.pth.tar')

        print('Save Successfully!')
        print('------------------------------------------------------------------------')


if __name__ == '__main__':
    global_time = time.time()
    if args.evaluate:
        test(start_epoch)
    else:
        for epoch in range(start_epoch, args.epochs):
            train(epoch)
            test(epoch)
    print(time.time()-global_time)





