import argparse
import os
import random
import shutil
import time
import datetime
import warnings
import sys

import numpy as np
import pandas as pd
from PIL import Image

# import nonechucks as nc

import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
 
from torchvision import transforms as trn
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from sklearn.metrics import accuracy_score


from functools import partial
import pickle

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

# python /media/Green01/data_zhangfan/code/tools/classification_multi/main.py ./evaluate_dataFrame.p -a=densenet121 -t=6 -c=10 -b=130 --gpu='0,1'   --topk=3  --resume=./2019-06-21_recent_checkpoint.pth.tar -p -e
# https://github.com/msamogh/nonechucks.git


# python main_perception.py ./pulse_score_cleaned.p -a=densenet121 -t=1 -c=10 -b=130 --gpu='0,1'   --topk=3 
#python test.py ./test-500-10cities.p -a=resnet18 -s=./here -b=30 -p --gpu='0,2' -c=10 -e --resume=./here/model_best.pth.tar
#python test.py ./test-500-10cities.p -a=resnet18 -s=./here -b=10 -p --gpu='0,2' -c='1,2,3,10' -e 

 
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

date_today = str(datetime.datetime.now().date()) 

parser = argparse.ArgumentParser(description='PyTorch classification Training')

parser.add_argument('data', metavar='DIR', 
                    help='dataframe path to dataset ' 'include: index path label split')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-t', '--num-tasks', required = True, type=int, metavar='N',
                    help='number of tasks for classification')

parser.add_argument('-c', '--num-classes',  nargs='+', type=int,
                    help='category number list for each task')
parser.add_argument('-w', '--class-weights',  nargs='+', type=int, default=None,
                    help='weights for each task')
parser.add_argument('-s', '--save-folder', default = './model', metavar='PATH', type=str,
                    help='folder path to save model files and log files (default: ./model)')
parser.add_argument('--only-best',  action='store_true',
                    help='only save best model')
parser.add_argument('-m', '--save-prefix', default = date_today, metavar='PATH', type=str,
                    help='prefix string to specify version or task type  (default: [date])')
parser.add_argument('-l', '--resume-log', default= '', metavar='PATH', type=str, 
                    help='path of log file to append (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--key-file', default='', type=str, 
                    help='path for classfication2regression (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',  action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--pretrained', dest='pretrained',  action='store_true',
                    help='use pre-trained model')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--safe-load',  action='store_true',
                    help='way to build data loader---handle decayed samples or not')
parser.add_argument('--img-size', default=[224,224], type=list,
                    help='size to resize image')
parser.add_argument('--lr-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)', dest='lr')
parser.add_argument('--lr-deGamma', default=0.1, type=float,
                    metavar='LR', help='lr-deGamma (default: 0.1)')
parser.add_argument('--lr-deStep', default=50, type=int,
                    metavar='LR', help='lr-deStep (default: 50)')  
    
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--topk', default=2, type=int,
                    help='print topk accuracy (default: 3)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default= '0', type=str,
                    help='GPU id(s) to use.'
                         'Format: a comma-delimited list')
best_acc1 = 0
    
def main():
    args = parser.parse_args()

    print(args)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if len(args.gpu) is not 0:
        print('You have chosen a specific GPU(s) : {}'.format(args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
        
    if (len(args.num_classes) == 1)&(args.num_tasks!=1):
        # if num of classes are identify for each class, take first input for all  
        args.num_classes = args.num_tasks * [int(args.num_classes[0])]
        
    elif len(args.num_classes) != args.num_tasks:
        print('Length of category list and number of tasks should be consistent')
        return
    else:
        args.num_classes = [int(x) for x in args.num_classes]
    

    print('Number of tasks: ', args.num_tasks)
    print('category for each task: ', args.num_classes)
    
    if not os.path.exists(args.save_folder) :
        os.makedirs(args.save_folder)
        
    main_worker(args)

def main_worker(args):
    global best_acc1

    model_urls = {
        'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    }
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        outList = []
        for i in range(args.num_tasks) :
            thisOut =  getattr(self, 'classifier' + str(i))(out)
            outList.append(thisOut)
        return outList

    newDenseNet = torchvision.models.densenet.DenseNet
    newDenseNet.forward = forward


    def densenet121(pretrained=False, **kwargs):
        r"""Densenet-121 model from
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = newDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                         **kwargs)
        if pretrained:
            # '.'s are no longer allowed in module names, but pervious _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(model_urls['densenet121'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            model.load_state_dict(state_dict)
        return model

    model = densenet121(pretrained =args.pretrained )
    classifierNameList = ['classifier' + str(_t) for _t in range(args.num_tasks)]

    for classifierName, num in zip(classifierNameList, args.num_classes) :
        module = nn.Linear(in_features=1024, out_features= num, bias = True )
        setattr(model, classifierName, module)

    model = torch.nn.DataParallel(model).cuda()
    

            
    # define loss function (criterion) and optimizer
    if args.class_weights == None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor(args.class_weights)).cuda()

    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    tf = trn.Compose([
            trn.Resize(args.img_size),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) #python 3

#             checkpoint = torch.load(args.resume) #python 2
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print("Current accuracy in validation: {}".format(checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return                
            
    if args.evaluate:
        if os.path.isfile(args.resume):
            evaluate( model, criterion, tf, args)
            return
        else:
            print('Evaluation mode without a pretrained model, quit.'    )

            return
    
    # Data loading code
    
    dataDF = pd.read_pickle(args.data)
    trainDF = dataDF[dataDF.split == 'train'].reset_index(drop=True)
    valDF = dataDF[dataDF.split == 'val'].reset_index(drop=True)
    
    args.batch_num_val = len(valDF)/args.batch_size
    args.batch_num_train = len(trainDF)/args.batch_size
    root = '/data_nas/huangyj/placePulse/rawData/googleviews-110802files/'

    trainDataset = Dataset(root+trainDF.name, trainDF.label, trainDF.task, tf)
    valDataset = Dataset(root+valDF.name, valDF.label, valDF.task, tf)

    if args.safe_load is True:
        
        trainDataset = nc.SafeDataset(trainDataset)
        train_loader = nc.SafeDataLoader(
            trainDataset,
            batch_size= args.batch_size, 
            shuffle=True,
            num_workers= args.workers, 
            pin_memory=True)
        
        valDataset = nc.SafeDataset(valDataset)
        val_loader = nc.SafeDataLoader(
            valDataset,
            batch_size= args.batch_size, 
            shuffle=True,
            num_workers= args.workers, 
            pin_memory=True)
    else:
        
        train_loader = torch.utils.data.DataLoader(
                trainDataset,
                batch_size= args.batch_size, 
                shuffle=True,
                num_workers= args.workers, 
                pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
                valDataset,
                batch_size= args.batch_size, 
                shuffle=False,
                num_workers= args.workers, 
                pin_memory=True)

        
    print('data loaded: ', str(datetime.datetime.now()))

    logList = []
    if len(args.resume_log):
        logList.append(pd.read_pickle(args.resume_log))
        
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)
       
        # train for one epoch
        logEpochDF = train(train_loader, model, criterion, optimizer, epoch, args)
        logList.append(logEpochDF)

        # evaluate on validation set
        logEpochDF, acc1s = validate(val_loader, model, criterion, epoch, args)
        logList.append(logEpochDF)
        
        acc1sMean = np.array(acc1s).mean()
        # remember best acc@1 and save checkpoint
        is_best = acc1sMean > best_acc1
        best_acc1 = max(acc1sMean, best_acc1)
        
        if args.only_best is True:
            save_model_name =  args.save_prefix +'_recent' + '_checkpoint.pth.tar'
        else:
            save_model_name =  args.save_prefix +'_' + str(epoch) + '_checkpoint.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_folder, save_model_name )
        
        save_log_name = args.save_prefix + '_log.p'
        pd.concat(logList).reset_index(drop=True).to_pickle(os.path.join(args.save_folder, save_log_name))
    
def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = [AverageMeter() for _t in range(args.num_tasks)]
    top1 = [AverageMeter() for _t in range(args.num_tasks)]
    top5 = [AverageMeter() for _t in range(args.num_tasks)]

    # switch to train mode
    model.train()

    logList = []
     
    for i, (input, target, task) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        task = np.array(task)

        # compute output
        output = model(input)


        lossList = []
        for _t in range(args.num_tasks):
            task_index  = np.argwhere(task==_t).flatten()
            thisOutput = output[_t][task_index,:]
            thisTarget = target[task_index]
            if len(thisTarget) == 0:
                continue
            loss = criterion(thisOutput, thisTarget)
            lossList.append(loss)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(thisOutput, thisTarget, topk=(1, args.topk))
            losses[_t].update(loss.item(), input.size(0))
            top1[_t].update(acc1[0], input.size(0))
            top5[_t].update(acc5[0], input.size(0))
            date_time = str(datetime.datetime.now()) 

            stepLogDict = {'time':date_time,
                           'type':'train',
                           'epoch':epoch,
                           'step':i, 
                           'task':_t,
                           'loss':float(losses[_t].val),
                           'top1':float(top1[_t].val),
                           'top5':float(top5[_t].val)}
            logList.append(stepLogDict)

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {date_time}\t'
                      'Task {task}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, args.batch_num_train, date_time= date_time, task = _t,
                          loss=losses[_t], top1=top1[_t], top5=top5[_t]))
            # compute gradient and do SGD step
        optimizer.zero_grad()
        sum(lossList).backward()
        optimizer.step()
        
    logEpochDF = pd.DataFrame(logList)
    return logEpochDF

def validate(val_loader, model, criterion, epoch, args):
    losses = [AverageMeter() for _t in range(args.num_tasks)]
    top1 = [AverageMeter() for _t in range(args.num_tasks)]
    top5 = [AverageMeter() for _t in range(args.num_tasks)]

    # switch to evaluate mode
    model.eval()
    logList = []
    with torch.no_grad():
        lossList = []
        for i, (input, target, task) in enumerate(val_loader):
           
            input = input.cuda()
            target = target.cuda()
            task = np.array(task)
            
            # compute output
            output = model(input)
                        
            for _t in range(args.num_tasks):
                task_index  = np.argwhere(task==_t).flatten()
                thisOutput = output[_t][task_index,:]
                thisTarget = target[task_index]
                if len(thisTarget) == 0:
                    continue
                loss = criterion(thisOutput, thisTarget)
                
                # measure accuracy and record loss
                acc1, acc5 = accuracy(thisOutput, thisTarget, topk=(1, args.topk))
                losses[_t].update(loss.item(), input.size(0))
                top1[_t].update(acc1[0], input.size(0))
                top5[_t].update(acc5[0], input.size(0))
                date_time = str(datetime.datetime.now()) 
    
                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {date_time}\t'
                          'Task {task}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i, args.batch_num_val, date_time= date_time, task = _t,
                              loss=losses[_t], top1=top1[_t], top5=top5[_t]))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1[_t], top5=top5[_t]))
        
        
        for _t in range(args.num_tasks):
            date_time = str(datetime.datetime.now())
            stepLogDict = {'time':date_time,
                           'type':'val',
                           'task':_t,
                           'epoch':epoch,
                           'step':None, 
                           'loss':float(losses[_t].avg),
                           'top1':float(top1[_t].avg),
                           'top5':float(top5[_t].avg)}
            logList.append(stepLogDict)

    logEpochDF = pd.DataFrame(logList)
    return logEpochDF, [top1[_t].avg for _t in range(args.num_tasks)]

def evaluate( model, criterion, transformer, args):
    
    evaluateDF = pd.read_pickle(args.data).reset_index(drop=True)
    for _t in range(args.num_tasks):
        evaluateDF['predict_' + str(_t)] = None 
        evaluateDF['prob_' + str(_t)] = None 
        evaluateDF['allprob_' + str(_t)] = None

    imglist = evaluateDF.path.tolist()
    indexlist = evaluateDF.index.tolist()

    dataset = EvaluateDataset(imglist, indexlist, transformer)
    loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False)
 
    # switch to evaluate mode
    model.eval()
     
    num_batches = len(evaluateDF) / args.batch_size
    
    with torch.no_grad():
        for batch_idx, (input, paths, img_indexes) in enumerate(loader):
 
            print(str(datetime.datetime.now()), ' %d / %d' % (batch_idx, num_batches))

            input = input.cuda()
            #input = torch.autograd.Variable(input, volatile = True).cuda()

            logits = model(input)
            for _t in range(args.num_tasks):
                
                h_x = torch.nn.functional.softmax(logits[_t], 1).data.squeeze()
                allProbs = h_x.cpu().numpy().round(3)
                allList = []
                [allList.append(allProbs[j]) for j in range(allProbs.shape[0])]

                probs, idx = h_x.sort(1, True)
                probs = probs[:,0].cpu().numpy()
                idx = idx[:,0].cpu().numpy()

                batchDF = pd.DataFrame({'prob_{}'.format(_t):probs, 'predict_{}'.format(_t):idx,'allprob_{}'.format(_t):allList })

                batchDF['path'] = paths
                batchDF.index = img_indexes

                evaluateDF.update(batchDF)
                
            if batch_idx%1000 == 0:
                save_path = os.path.join(args.save_folder, args.save_prefix + '_{}'.format(batch_idx) +'_evaluate.p')
                print('batch size: ', args.batch_size, 'Evaluation file was saved at ', save_path)
                evaluateDF.to_pickle(save_path)
        save_path = os.path.join(args.save_folder, args.save_prefix + '_{}'.format(batch_idx) +'_evaluate.p')
        print('batch size: ', args.batch_size, 'Evaluation file was saved at ', save_path)
        evaluateDF.to_pickle(save_path)
    if args.key_file != '':
        print('convert label to score')
        c2rKEYDF = pd.read_pickle(args.key_file)
        midsList = c2rKEYDF.mids.values
        scalerList = c2rKEYDF.scaler.values

        for i in range(6):
            #_interList = (evaluateDF['allprob_{}'.format(i)]*midsList[i]).apply(lambda x: sum(x) )
            _interList = evaluateDF['allprob_{}'.format(i)].apply(lambda x: sum(x*midsList[i]))
            deNormed = scalerList[i].inverse_transform(_interList) 
            evaluateDF['score_{}'.format(i)] = deNormed
    
    save_path = os.path.join(args.save_folder, args.save_prefix + '_all_evaluate.p')
    print('Evaluation file was saved at ', save_path)
    evaluateDF.to_pickle(save_path)
    
    if 'split' in evaluateDF.columns:
        trainDF = evaluateDF[evaluateDF.split =='train']
        valDF = evaluateDF[evaluateDF.split =='val']
        for _t in range(args.num_tasks):
            taskDF = evaluateDF[evaluateDF.task == _t]
            taskTrainDF = trainDF[trainDF.task == _t]
            taskValDF = valDF[valDF.task == _t]
            totalAcc = accuracy_score(taskDF.label.tolist(), taskDF['predict_' + str(_t) ].tolist())
            trainAcc = accuracy_score(taskTrainDF.label.tolist(), taskTrainDF['predict_' + str(_t) ].tolist())
            valAcc = accuracy_score(taskValDF.label.tolist(), taskValDF['predict_' + str(_t) ].tolist())

            print('Task: ', _t)
            print('Acc train: ', trainAcc)
            print('Acc val: ', valAcc)
            print('Acc total: ', totalAcc)

    
    
    
#===================== Utils =====================#


class Dataset(data.Dataset):
    def __init__(self, imgList, labelList, taskList, transform = None):
     
        self.imgList = imgList
        self.labelList = labelList
        self.taskList = taskList
        self.transform = transform
            
    def __getitem__(self, index):

        img_path = self.imgList[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labelList[index]
        task = self.taskList[index]
        if self.transform:
            image = self.transform(image)
        sample = (image,label,task)
        
        return sample
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgList) 
    
class EvaluateDataset(data.Dataset):
    def __init__(self, imgList, imgindexList, transform = None):
     
        self.imgList = imgList
        self.transform = transform
        self.imgindexList = imgindexList
        
    def __getitem__(self, index):

        img_path = self.imgList[index]
        img_index = self.imgindexList[index]
        
        image = Image.open(img_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
        
        return image,img_path, img_index
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgList) 
    
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

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_deGamma ** (epoch // args.lr_deStep))
 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
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

def save_checkpoint(state, is_best, folder = './', filename='checkpoint.pth.tar'):
    save_path = os.path.join(folder, filename)
    torch.save(state,  save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(folder, 'model_best.pth.tar')  )

    
if __name__ == '__main__':
    main()
