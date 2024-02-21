## 已通过，可以在ubuntu20.04上运行
import os
import math
import random
import argparse  #命令项选项
from time import time
import glob   #查找文件目录
import sys
from pathlib import Path
from typing import Iterable,Optional

import matplotlib
import numpy as np
import torch
import torch.multiprocessing #相同数据的不同进程中共享视图。
import torch.utils.data
#torch.multiprocessing.set_sharing_strategy('file_system')

import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import timm
from timm.utils import accuracy
from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@torch.no_grad()
def evaluate(data_loader, model, device):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'
    # seitch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        #将数据传入到设备
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        output = model(images)
        #计算损失
        loss = criterion(output, target)
        # output = torch.nn.functional.softmax(output, dim=-1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        # print(batch_size)
        # 更新到log中
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gater the stats from all processes
    metric_logger.synchronize_between_processes()
    print('*Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler ,max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    print_freq = 2
    #梯度更新跨度
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (samples, targets) in enumerate(data_loader):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)
        # warmup_lr = args.lr*(min(1.0,epoch/2.))
        # 设定学习率
        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr
        #计算损失
        loss = criterion(outputs, targets)
        loss /= accum_iter
        #完成梯度更新
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step+1) % accum_iter == 0)
        loss_value = loss.item()
        if (data_iter_step+1) % accum_iter == 0:
            optimizer.zero_grad()

        if not math.isfinite(loss_value):
            print("Loss is {},stopping training".format(loss_value))
            sys.exit(1)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step/len(data_loader)+epoch)*1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('ls', warmup_lr, epoch_1000x)
            print(f"Epoch: {epoch},Step: {data_iter_step},Loss: {loss}, Lr: {warmup_lr}")



def build_transform(is_train,args):
    if is_train:
        print("train transform")
        return torchvision.transforms.Compose([
            #规整图片大小
            torchvision.transforms.Resize((args.input_size,args.input_size)),
            #随机垂直翻转
            torchvision.transforms.RandomHorizontalFlip(),
            #随机水平翻转
            torchvision.transforms.RandomVerticalFlip(),
            #随机调整角度
            torchvision.transforms.RandomPerspective(distortion_scale=0.6,p=1.0),
            #随机加入高斯噪声
            torchvision.transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
            #转为张量【0,1】
            torchvision.transforms.ToTensor(),
        ]
        )
    print("eval transform")
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size,args.input_size)),
            torchvision.transforms.ToTensor(),
        ]
    )

def build_dataset(is_train,args):
    #进行图像变换预处理
    transform = build_transform(is_train, args)
    #根据训练和测试的需要获取训练集和测试集的路径
    path = os.path.join(args.root_path, 'train' if is_train else 'test')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)
    #查看数据类别及其索引
    print(f"finding classes from {path}:\t{info[0]}")
    print(f"mapping classes from {path} to indexes:\t{info[1]}")
    return dataset
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre_training', add_help=False)
    parser.add_argument('--batch_size', default=12, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints')
    # Model parameters
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float,default=0.0001,metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--root_path', default='AllData',
                        help='path where to save,empty for no saving')
    parser.add_argument('--output_dir', default='./output_dir_pretrained',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pretrained',
                        help='path where to tensorboard log')
    parser.add_argument('--resume', default='checkpoint-95.pth',
                        help='resume from checkpoint')
    # parser.add_argument('--resume', default='',
    #                     help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=5, type= int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_men=True)
    return parser
def main(args,mode,test_image_path=''):
    print(f"{mode} mode...")
    if mode == 'train':
        # 构建数据批次
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)
        #对训练数据打散（随机采样）
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        #对训练集顺序采样
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last =False
        )
        # 构建模型resnet50
        model =timm.create_model('resnet50', pretrained=True, num_classes=49, drop_rate=0.1, drop_path_rate=0.1)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params(M): %.2f' % (n_parameters/1.e6))
        #定义交叉熵目标（损失）函数
        criterion = torch.nn.CrossEntropyLoss()
        #调用优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #创建记录日志文件夹
        os.makedirs(args.log_dir, exist_ok=True)
        #记录loss曲线
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scaler = NativeScaler()
        # 读入已有的模型
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

        for epoch in range(args.start_epoch, args.epochs):
            print(f"Epoch {epoch}")
            print(f"length of data_loader_train is {len(data_loader_train)}")

            if epoch % 1 == 0:
                print("Evaluating...")
                model.eval()
                test_stats = evaluate(data_loader_val, model, device)
                #打印测试结果
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

                if log_writer is not None:
                    log_writer.add_scalar('perr/test_acc1',test_stats['acc1'], epoch)
                    log_writer.add_scalar('perr/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perr/test_loss', test_stats['loss'], epoch)
                model.train()
            print("Training...")
            train_stats = train_one_epoch(
                model,criterion,data_loader_train,
                optimizer,device,epoch+1,
                loss_scaler, None,
                log_writer=log_writer,
                args=args
            )
            if args.output_dir:
                print("Saving checkpoints...")
                #保存模型
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch
                )
    else:
        model = timm.create_model('resnet50', pretrained=True, num_classes=49, drop_rate=0.1, drop_path_rate=0.1)
        class_dict = {'卷心菜': 0, '土豆': 1, '大蒜': 2, '大豆': 3, '山竹': 4, '山莓': 5, '彩椒': 6, '杏桃': 7, '杨桃': 8, '柚子': 9, '柠檬': 10, '柿子': 11, '栗子': 12, '核桃': 13, '桑椹': 14, '梨': 15, '椰子': 16, '榛子': 17, '樱桃': 18, '橘子': 19, '油桃': 20, '洋葱': 21, '火龙果': 22, '牛油果': 23, '猕猴桃': 24, '玉米': 25, '甜菜': 26, '生姜': 27, '生菜': 28, '番石榴': 29, '百香果': 30, '石榴': 31, '红浆果': 32, '胡萝卜': 33, '芒果': 34, '花椰菜': 35, '苹果': 36, '茄子': 37, '草莓': 38, '荔枝': 39, '菠萝': 40, '葡萄': 41, '蓝莓': 42, '西瓜': 43, '西红柿': 44, '豌豆': 45, '金橘': 46, '香蕉': 47, '黄瓜': 48}
            #n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        loss_scaler = NativeScaler()
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
        model.eval()

        image = Image.open(test_image_path).convert('RGB')
        image1 = image.resize((args.input_size,args.input_size),Image.ANTIALIAS)
        image = torchvision.transforms.ToTensor()(image1).unsqueeze(0)

        with torch.no_grad():
            output = model(image)

        output = torch.nn.functional.softmax(output, dim=-1)
        #计算哪一个类别的概率最大
        class_idx = torch.argmax(output, dim=1)[0]
        #最大概率
        score = torch.max(output,dim=1)[0][0]
        #打印结果
        print(f"image path is {test_image_path}")
        print(f"score is {score.item()}, class id is {class_idx.item()}, class name is {list(class_dict.keys())[list(class_dict.values()).index(class_idx)]}")
        plt.subplot(1, 2, 1), plt.imshow(image1)
        plt.title(test_image_path), plt.axis('off')
        plt.subplot(1, 2, 2), plt.imshow(image1)
        plt.title(list(class_dict.keys())[list(class_dict.values()).index(class_idx)]+f"(预测） 得分 {score.item():.4f}"), plt.axis('off')
        plt.show()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    mode = 'test'
    if mode == 'train':
        print(device)
        print("train")
        main(args, mode = mode)
    else:
        print(matplotlib.matplotlib_fname())
        # images = glob.glob('卷心菜.jpg')#测试集路径
        images = glob.glob('./AllData/test/*/*.jpg')  # 测试集路径
        for image in images:
            print('\n')
            main(args, mode=mode, test_image_path=image)
