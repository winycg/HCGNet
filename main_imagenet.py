'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

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
parser.add_argument('--train_data', default='./data/ImageNet/train/', type=str, help='train data location')
parser.add_argument('--val_data', default='./data/ImageNet/val/', type=str, help='validation data location')
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
parser.add_argument('--print_freq', type=int, default=100)


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(epoch):
    train_sampler.set_epoch(epoch)
    batch_times = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    end = time.time()
    start_time = time.time()
    start_batch_time = time.time()

    lr = adjust_lr(optimizer, epoch, args.init_lr * float(args.batch_size * args.world_size) / 256)

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

        if args.local_rank == 0:
            print('Epoch: {0}\tBatch: {1}\t Time {2:.3f}'.
                  format(epoch, i, time.time() - start_batch_time))

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            # to_python_float incurs a host<->device sync
            losses.update(float(reduced_loss.item()), inputs.size(0))
            top1.update(float(prec1.item()), inputs.size(0))
            top5.update(float(prec5.item()), inputs.size(0))

            batch_times.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(trainloader),
                    args.world_size * args.batch_size / batch_times.val,
                    args.world_size * args.batch_size / batch_times.avg,
                    batch_time=batch_times,
                    loss=losses, top1=top1, top5=top5))

        inputs, targets = prefetcher.next()
        start_batch_time = time.time()

    torch.cuda.empty_cache()
    if args.local_rank == 0:
        print('Epoch:{0}\t lr:{1:.6f}\t duration:{2:.3f}'
              .format(epoch, lr, time.time() - start_time))
        with open('result/' + os.path.basename(__file__).split('.')[0] + '.txt', 'a+') as f:
            f.write(str(time.time()-start_time))

def test(epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    end = time.time()
    global best_acc

    prefetcher = data_prefetcher(testloader)
    inputs, targets = prefetcher.next()

    with torch.no_grad():
        i = 0
        while inputs is not None:
            i += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            reduced_loss = reduce_tensor(loss)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            losses.update(float(reduced_loss.item()), inputs.size(0))
            top1.update(float(prec1.item()), inputs.size(0))
            top5.update(float(prec5.item()), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {2:.3f} ({3:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(testloader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

            inputs, targets = prefetcher.next()
    torch.cuda.empty_cache()

    if args.local_rank == 0:
        with open('result/' + os.path.basename(__file__).split('.')[0] + '.txt', 'a+') as f:
            f.write(',' + str(top1.avg) + ' ' + str(top5.avg) + '\n')
        print('Test top1 accuracy: ', top1.avg)
        print('Test top5 accuracy: ', top5.avg)
        print('Test loss: ', losses.avg)

        is_best = False
        if best_acc < top1.avg:
            best_acc = top1.avg
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    global_time = time.time()
    if args.evaluate:
        test(start_epoch)
    else:
        for epoch in range(start_epoch, args.epochs):
            train(epoch)
            test(epoch)
    print(time.time()-global_time)





