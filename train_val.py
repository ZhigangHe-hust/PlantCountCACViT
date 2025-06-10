import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
import math
import sys
from PIL import Image
from thop import profile

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147_384 import TransformTrain # 数据处理函数
import models.CACViT as CntVit
from val import val_func

# 获取命令行参数，若命令行没有设置参数，则返回默认(default)参数
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)

    # 梯度累积步数，用于在内存不足时模拟更大的批次（如 batch_size=4 + accum_iter=4 等效于 batch_size=16）
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters:
    # 1.模型名称（如 mae_vit_base_patch16_dec512d8b）
    parser.add_argument('--model', default='mae_vit_base_patch16_dec512d8b', type=str, metavar='MODEL',
                        help='Name of model to train')
    # 2.图像块（patches）的遮盖比例（MAE核心参数）
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    # 3.是否对每个图像块的像素值进行归一化后再计算损失，默认关闭
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters:
    # 1.权重衰减系数（L2正则化强度）
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # 2.直接指定的绝对学习率
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    # 3.基础学习率，实际学习率由 blr * total_batch_size / 256 计算得出
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # 4.学习率调度器的最低学习率（如余弦退火）
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    # 学习率预热轮次（逐步增加学习率以避免训练初期不稳定）
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters:
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    
    # TensorBoard 日志目录
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=314, type=int)

    # 从指定检查点文件恢复训练
    parser.add_argument('--resume', default='./pretrain/best_model_of_CACViT.pth',
                        help='resume from checkpoint')
    
    # 起始训练轮次（配合 --resume 使用）
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', 
                        help='start epoch')
    
    # 数据加载的子进程数（建议设为CPU核心数）
    parser.add_argument('--num_workers', default=10, type=int)

    # 是否固定内存（pin_memory=True 可加速GPU数据传输，默认开启）
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # 1.分布式训练的进程总数
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # 2.当前进程的本地Rank（由启动脚本自动分配）
    parser.add_argument('--local_rank', default=-1, type=int)

    parser.add_argument('--dist_on_itp', action='store_true')
    # 3.分布式训练的初始化URL（如 env:// 表示从环境变量获取）
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

# 强制 ​​每个 CUDA 内核同步执行​​，即 CPU 必须等待当前 GPU 操作完成后再继续。
# 产生的错误会 ​​立即抛出​​，并准确指向触发问题的代码行，极大简化调试过程
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

data_path = './data'
anno_file = './data/annotations.json'
data_split_file = './data/Train_Val_Test.json' # 划分的测试集，训练集，验证集信息
im_dir = './data/images'
gt_dir = './data/gt_density_maps'
class_file = './data/ImageClasses.txt'

# 解析文件对象 f 中的 JSON 数据​​，将其转换为 Python 对象（通常是字典）
with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

class_dict = {}
with open(class_file) as f:
    for line in f:
        key = line.split()[0]
        val = line.split()[1:]
        class_dict[key] = val

# 得到处理后的训练数据
class TrainData(Dataset):
    def __init__(self):
        
        self.img = data_split['train'] # self.img是一个列表
        random.shuffle(self.img)
        self.img_dir = im_dir  # 图片文件路径

    def __len__(self):
        return len(self.img) # 返回训练图片的数量

    def __getitem__(self, idx):
        im_id = self.img[idx]   # 返回图片的名称，如"abelmoschus_esculentus_green_fruits_1"
        anno = annotations[im_id]   # 对应图片的标注，是一个字典
        bboxes = anno['box_examples_coordinates'] # 返回标注框的坐标信息
        # [ [[106, 511], [179, 511], [179, 592], [106, 592]], [[434, 602], [505, 602], [505, 676], [434, 676]], 
        #   [[387, 216], [457, 216], [457, 282], [387, 282]] ], 

        rects = list() # 用来存储边界框左上顶点和右下顶点坐标信息的列表
        for bbox in bboxes:
            # bbox:[[106, 511], [179, 511], [179, 592], [106, 592]]
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])
            # rects:[ [511,106,592,179],··· ]

        dots = np.array(anno['points']) 

        image = Image.open('{}/{}'.format(im_dir, im_id)) # 拼接路径并打开图片，返回一个Image对象，
        image.load() # 强制将图片的像素数据加载到内存
        density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy" # 得到图片对应的密度图文件路径
        density = np.load(density_path).astype('float32') 
        m_flag = 0  
        # np.load()​​：
        # 加载 .npy 文件（NumPy 的二进制格式），返回一个 ​​NumPy 数组​​。
        # ​​.astype('float32')​​：
        # 将数组数据类型强制转换为 float32（单精度浮点数），这是深度学习中常用的数据类型（节省内存且兼容GPU计算）。
        # ​​典型密度图数据​​：
        # 密度图是单通道矩阵，每个像素值表示该位置的物体密度（如人群计数任务中，值越高表示人越密集）。
        # 示例密度图形状：(height, width) 或 (1, height, width)

        sample = {'image':image,'lines_boxes':rects,'gt_density':density, 'dots':dots, 'id':im_id, 'm_flag': m_flag} # 样本字典
        sample = TransformTrain(sample)  # 调用util.FSC147_384.py中的TransformTrain函数，对训练数据进行处理
        return sample['image'], sample['gt_density'], sample['boxes'], sample['m_flag'], sample['scale']

# 主函数
def main(args):
    # 分布式训练模式的初始化​，并打印初始化信息（仅主进程可见）
    misc.init_distributed_mode(args)
    # 输出当前 Python 脚本所在的绝对路径（不包括脚本文件名）
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # 将 args（通常是 argparse.Namespace 对象）的所有参数按 ​​每行一个参数​​ 的格式输出
    print("{}".format(args).replace(', ', ',\n'))

    # 设备、随机种子、训练数据集
    device = torch.device(args.device) # 设备
    seed = args.seed + misc.get_rank() # 修复种子以实现可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True # 启用 cuDNN（NVIDIA 的深度学习加速库）的​​自动优化模式​​
    dataset_train = TrainData() # 定义训练数据
    print(dataset_train)

    # 分布式数据采样器设置​
    if True:  # args.distributed:
        num_tasks = misc.get_world_size() # 总进程数（GPU数量×节点数）
        global_rank = misc.get_rank()     # 当前进程的全局排名（0为主进程）
        # 创建分布式采样器​
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else: #非分布式备用方案​
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # 日志记录器设置​
    if global_rank == 0 and args.log_dir is not None:    # 仅主进程创建日志目录​
        os.makedirs(args.log_dir, exist_ok=True)         # 创建日志目录（如果不存在）
        log_writer = SummaryWriter(log_dir=args.log_dir) # 初始化TensorBoard写入器​
    else:
        log_writer = None

    # 数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # 模型
    model = CntVit.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # 只有base_lr是指定的
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed: # 如果是分布式训练的情况，将模型部署到各个设备
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 将偏差层(bias)和范数层(norm layers)的 wd 设置为 0
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # 优化器设置
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    loss_scaler = NativeScaler()# 混合精度训练工具
    min_MAE = 99999
    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)
    
    # 开始训练
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    mae=10000000
    mse=10000000
    # ----------------------------------------for_1-----------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)# 在分布式训练中，通过调用 sampler.set_epoch(epoch)，​​强制每个epoch的数据划分不同
        
        model.train(True) # 启用模型的训练模式（开启Dropout/BatchNorm等训练专用层）
        metric_logger = misc.MetricLogger(delimiter="  ") # 创建一个日志记录工具，用于跟踪训练过程中的各项指标（如loss、准确率等）
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')) # 添加学习率记录器​，专门记录学习率（lr）的变化，格式化为6位小数
        header = 'Epoch: [{}]'.format(epoch) # 在训练日志中标识当前epoch，便于监控进度
        print_freq = 20 # 每处理20个batch打印一次日志
        accum_iter = args.accum_iter # 梯度累积参数​，每accum_iter个batch才更新一次模型参数

        # 优化器梯度清零
        optimizer.zero_grad() # 优化器梯度清零(将记录梯度的矩阵清零,以便于反向传播时计算梯度并将梯度值存储在梯度矩阵中)

        # 如果存在TensorBoard日志记录器（log_writer），则打印其日志目录路径
        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))

        # 在每个epoch开始时重置统计量
        train_mae = 0            # MAE分子: Σ|y_pred - y_true|
        train_rmse = 0           # RMSE分子: Σ(y_pred - y_true)^2
        train_rmae = 0           # rMAE分子：Σ(|y_pred - y_true|/|y_true|)
        pred_cnt = 0             # 模型预测的​​目标总数​​
        gt_cnt = 0               # 标注的​​真实目标总数​​
        total_gt = 0.0           # 所有真实值的和
        total_gt_squared = 0.0   # 所有真实值平方和
        sample_flops = 0          # 一个batch的flops
        total_flops = 0
        flops_per_sample = []
        
        # ----------------------------------------for_2-----------------------------------------
        for data_iter_step, (samples, gt_density, boxes, m_flag, scales) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
            
            # 周期性调整学习率，每次达到梯度累积周期时（data_iter_step % accum_iter == 0）调整学习率
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

            # 数据预处理：1.non_blocking=True：异步数据传输（提高效率） 2.half()：转换为FP16半精度（减少显存占用）
            samples = samples.to(device, non_blocking=True).float()
            gt_density = gt_density.to(device, non_blocking=True).float()
            boxes = boxes.to(device, non_blocking=True).float()
            scales = scales.to(device, non_blocking=True).float()

            # 采样策略：选择每张图像使用的目标实例数（固定为3-shot），截取相应数量的边界框和尺度信息
            flag = 0
            for i in range(m_flag.shape[0]):
                flag += m_flag[i].item()
            if flag == 0:
                shot_num = random.randint(0,3)
            else:
                shot_num = random.randint(1,3)
            shot_num = 3 # 每张图像使用的目标实例数实际设置为固定值
            boxes = boxes[:,:shot_num,:,:,:]
            scales = scales[:,:shot_num,:]

            # 模型前向传播
            with torch.cuda.amp.autocast():
                inputx = [samples,boxes,scales]
                output = model(inputx)

            # 计算当前批次FLOPs
            with torch.no_grad():
                macs, _ = profile(model_without_ddp, inputs=(inputx,), verbose=False)
            sample_flops += macs

            # 记录当前批次的FLOPs
            flops_per_sample.append(sample_flops / 1e9)  # 转换为GFLOPs
            total_flops += sample_flops

            ''' 
            损失计算（独特设计）
            1.​创新点​​：
                创建随机二值掩码(80%为1,20%为0)
                ​​目的​​:稀疏计算损失，降低计算量
                掩码尺寸:384x384 → 对应输入分辨率
            2.​损失计算​​：
                使用L2损失:(预测密度 - 真实密度)^2
                只计算被掩码覆盖区域的损失(降低80%计算量)
                归一化:除以总像素数和batch大小
            '''
            mask = np.random.binomial(n=1, p=0.8, size=[384,384])
            masks = np.tile(mask,(output.shape[0],1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(device)
            loss = (output - gt_density) ** 2
            loss = (loss * masks / (384*384)).sum() / output.shape[0]
            loss_value = loss.item()
            if loss_value<10 == False :
                print(loss_value.dtype)

            # 评估指标计算
            batch_mae = 0
            batch_rmse = 0
            batch_rmae = 0
            batch_total_gt = 0
            batch_total_gt_squared = 0
            pred_cnt_list = []
            gt_cnt_list = []
            output_list = []

            for i in range(output.shape[0]): # output.shape[0]==16，因为一个批次有16张图片
                pred_cnt = torch.sum(output[i]/60).item()  # 预测的目标数量
                pred_cnt_list.append(pred_cnt)
                output_list.append(output[i])
                gt_cnt = torch.sum(gt_density[i]/60).item() # 实际目标数量
                gt_cnt_list.append(gt_cnt)

                cnt_err = abs(pred_cnt - gt_cnt)            # 计数结果误差
                batch_mae += cnt_err
                batch_rmse += cnt_err ** 2
                if gt_cnt != 0:
                    batch_rmae += cnt_err / gt_cnt
                else:
                    # 当没有真实标注样本时，直接加0.0
                    batch_rmae += 0
                batch_total_gt += gt_cnt
                batch_total_gt_squared += gt_cnt**2
                
                if i == 0 : # 打印样本的损失和样本第1张图片的预测数量，实际数量，计数结果误差，样本总计数结果误差，样本总计数结果均方误差
                    print(f'{data_iter_step}/{len(data_loader_train)}: loss: {loss_value},  pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {abs(pred_cnt - gt_cnt)},  AE: {cnt_err},  SE: {cnt_err ** 2}, {shot_num}-shot, FLOPs: {sample_flops/1e9:.2f} GFLOPs')
            train_mae += batch_mae
            train_rmse += batch_rmse
            train_rmae += batch_rmae
            total_gt += batch_total_gt
            total_gt_squared += batch_total_gt_squared

            '''        
            TensorBoard可视化
              可视化内容​​：
                边界框:boxes[0]
                真实密度图：叠加在原始图像上
                预测密度图：单独展示
                预测密度与图像叠加效果
            '''
            if log_writer is not None and data_iter_step == 0:
                fig = output[0].unsqueeze(0).repeat(3,1,1)
                f1 = gt_density[0].unsqueeze(0).repeat(3,1,1)
                log_writer.add_images('bboxes', (boxes[0]), int(epoch),dataformats='NCHW')
                log_writer.add_images('gt_density', (samples[0]/2+f1/10), int(epoch),dataformats='CHW')
                log_writer.add_images('density map', (fig/20), int(epoch),dataformats='CHW')
                log_writer.add_images('density map overlay', (samples[0]/2+fig/10), int(epoch),dataformats='CHW')
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            '''
            梯度更新
              梯度累积策略​​：
                loss /= accum_iter:平均累积的梯度
                loss_scaler:混合精度梯度缩放器
                当完成累积步数时((iter+1)%accum_iter==0):
                update_grad=True:执行参数更新
                重置梯度:optimizer.zero_grad()
            '''
            loss /= accum_iter  # 梯度累积归一化
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            torch.cuda.synchronize()

            '''
            训练指标记录与同步
            '''
            metric_logger.update(loss=loss_value)
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 3:
                """
                TensorBoard日志:
                X轴:epoch_1000x(统一不同batch大小的训练曲线)
                记录损失、学习率、MAE、RMSE
                """
                epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_scalar('MAE', batch_mae/args.batch_size, epoch_1000x)
                log_writer.add_scalar('RMSE', (batch_rmse/args.batch_size)**0.5, epoch_1000x)
        # ----------------------------------------for_2-----------------------------------------
        
        # ======= 新增：聚合分布式训练的指标 =======
        if args.distributed:
            # 跨GPU聚合指标
            train_mae = torch.tensor(train_mae).to(device)
            train_rmse = torch.tensor(train_rmse).to(device)
            total_gt = torch.tensor(total_gt).to(device)
            total_gt_squared = torch.tensor(total_gt_squared).to(device)
            
            # 求和操作
            torch.distributed.all_reduce(train_mae, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(train_rmse, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_gt, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_gt_squared, op=torch.distributed.ReduceOp.SUM)
            
            # 转换回Python标量
            train_mae = train_mae.item()
            train_rmse = train_rmse.item()
            total_gt = total_gt.item()
            total_gt_squared = total_gt_squared.item()
        
        # 过拟合测试时只使用一个批次的训练数据
        metric_logger.synchronize_between_processes()# 在分布式训练环境下，同步所有GPU进程之间的指标记录器
        print("Averaged stats:", metric_logger) # 打印整个训练过程中所有进程平均后的训练统计指标
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} # 将metric_logger中的指标转换为全局平均值的字典

        # 保存训练状态和模型
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):# 每10个epoch保存一次,训练结束前的最后一步保存
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        
        if args.output_dir and train_mae/(len(data_loader_train) * args.batch_size) < min_MAE:# 检查当前epoch的训练MAE是否是迄今为止最好的
            min_MAE = train_mae/(len(data_loader_train) * args.batch_size)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=666) # 保存当前最佳模型：checkpoint-666.pth
        
        if args.output_dir and epoch >= 20 and epoch % 1 == 0: # 训练超过30个epoch后，每2个epoch进行一次验证
            # 在验证集上评估当前模型
            mae_new,mse_new = val_func(model=model,device=device)

            # 如果当前MAE优于历史最佳,保存最佳模型
            if mae_new < mae:
                mae = mae_new
                mse = mse_new
                misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch+1)
                # 在测试集上面评估当前模型
                mae_test,mse_test = val_func(model=model,device=device,dataset='test')
                # 记录测试日志并写入log_test.txt文件
                log_stats =  {'TEST MAET': mae_test,
                              'TEST RMSE':  mse_test,
                              'epoch': epoch,}
                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log_test.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
            
            # 记录验证日志并写入log_val.txt文件
            log_stats =  {'VAL MAE': mae_new,
                          'VAL RMSE':  mse_new,
                          'BEST MAE': mae,
                          'BEST MSE': mse,
                          'epoch': epoch,}
            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log_val.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            model.train()
        
        n = len(data_loader_train) * args.batch_size # 样本总数
        m = total_gt_squared - 2 * (total_gt/n) * total_gt + n * (total_gt/n)**2 # R2的分母
        
        print('MAE:{:5.2f},RMSE:{:5.2f},rMAE:{:5.2f},rRMSE:{:5.2f},R2:{:5.2f}'.format(train_mae/n,(train_rmse/n)**0.5,train_rmae/n,((train_rmse/n)**0.5)/(total_gt/n),1-(train_rmse/m)))
        
        # 打印FLOPs统计结果
        # ===================================================================
        print("\nFLOPs Summary:")
        print(f"Total FLOPs for all samples: {total_flops/1e12:.4f} TFLOPs")
        print(f"Average FLOPs per sample: {total_flops/n/1e9:.4f} GFLOPs")
        print(f"Min sample FLOPs: {min(flops_per_sample)/args.batch_size:.4f} GFLOPs")
        print(f"Max sample FLOPs: {max(flops_per_sample)/args.batch_size:.4f} GFLOPs")
        print(f"Median sample FLOPs: {np.median(flops_per_sample)/args.batch_size:.4f} GFLOPs")
        # ===================================================================

        # 输出日志状态
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'MAE': train_mae/(len(data_loader_train) * args.batch_size),
                        'RMSE': (train_rmse/(len(data_loader_train) * args.batch_size))**0.5,
                        'rMAE': train_rmae/n,
                        'rRMSE': ((train_rmse/n)**0.5)/(total_gt/n),
                        'R2': 1-(train_rmse/m),
                        'FLOPs': total_flops/n,
                        'epoch': epoch,}

        # 将训练结果写入log.txt文件
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    # ----------------------------------------for_1-----------------------------------------

    # 打印训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 设置环境变量CUDA_VISIBLE_DEVICES为 '7'，限制程序仅使用编号为7的GPU设备。常用于多GPU环境中指定特定GPU运行代码
    args = get_args_parser() 
    args = args.parse_args()# 解析命令行输入的参数，结果存储在args变量中
    if args.output_dir:
        # 使用 pathlib.Path 创建输出目录。parents=True 允许自动创建父目录，exist_ok=True 表示目录已存在时不报错
        Path(args.output_dir).mkdir(parents=True, exist_ok=True) 
    main(args)# 调用主函数 main()，将解析后的参数 args 传递给它