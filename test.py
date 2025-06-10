import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import random
from thop import profile

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
import models.CACViT as CntViT

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/log_dir',
                        help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./pretrain/best_model_of_CACViT.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

data_path = './data'
anno_file = './data/annotations.json'
data_split_file = './data/Train_Val_Test.json' # 划分的测试集，训练集，验证集信息
im_dir = './data/images'
gt_dir = './data/gt_density_maps'
class_file = './data/ImageClasses.txt'

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

class TestData(Dataset):
    def __init__(self):
        self.img = data_split['test']
        self.img_dir = im_dir

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        # [ [[106, 511], [179, 511], [179, 592], [106, 592]], [[434, 602], [505, 602], [505, 676], [434, 676]], 
        #   [[387, 216], [457, 216], [457, 282], [387, 282]] ]

        dots = np.array(anno['points'])
        # [ [88, 177], [109, 236], [144, 273], [75, 350], [93, 671], [149, 567],
        #   [227, 690], [285, 647], [343, 521], [369, 440], [305, 341], [295, 291], 
        #   [263, 221], [301, 172], [375, 222], [409, 254], [509, 230], [485, 512], 
        #   [532, 533], [595, 499], [561, 270], [609, 203], [593, 302], [597, 351], 
        #   [678, 646], [812, 586], [695, 446], [773, 265], [394, 686], [463, 622], 
        #   [517, 632], [547, 642], [591, 640], [728, 581], [809, 660], [853, 584], [918, 560] ] 

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load() 
        W, H = image.size
        
        # 计算调整后的尺寸（16的整数倍），神经网络常要求输入尺寸为整数倍（如16）
        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        scale_factor = float(new_W)/ W # 宽度缩放比例

        # 数据预处理
        image = transforms.Resize((new_H, new_W))(image)# 图像尺寸调整
        Normalize = transforms.Compose([transforms.ToTensor()])# 转换为张量
        image = Normalize(image)# 像素值归一化到[0,1]

        rects = list()
        for bbox in bboxes:
            # 提取并缩放坐标（仅缩放x方向，y方向保持不变）
            x1 = int(bbox[0][0]*scale_factor)
            y1 = bbox[0][1]
            x2 = int(bbox[2][0]*scale_factor)
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        boxes = list()
        scale_x = []    # 宽度缩放因子
        scale_y = []    # 高度缩放因子
        cnt = 0         # 计数器
        for box in rects:
            cnt+=1
            if cnt>3:# 只处理前3个边界框
                break
            box2 = [int(k) for k in box]# 确保坐标整数化
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]

            # 计算相对缩放因子（除以384）
            scale_x1 = torch.tensor((x2-x1+1)/384)
            scale_x.append(scale_x1)
            scale_y1 = torch.tensor((y2-y1+1)/384)
            scale_y.append(scale_y1)

            # 裁剪边界框区域
            bbox = image[:,y1:y2+1,x1:x2+1]

            # 统一缩放到64x64
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy()) # 临时转NumPy便于处理
        
        # 合并缩放因子
        scale_xx = torch.stack(scale_x).unsqueeze(-1)# [N,1]
        scale_yy = torch.stack(scale_y).unsqueeze(-1)# [N,1]
        scale = torch.cat((scale_xx,scale_yy),dim=1)# [N,2] 组合为(w_scale, h_scale)
         # 转换boxes为Tensor
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

        # Only for visualisation purpose, no need for ground truth density map indeed.
        gt_map = np.zeros((image.shape[1], image.shape[2]),dtype='float32')# 初始化密度图（全零）
        # 标记物体中心点位置
        for i in range(dots.shape[0]):
            gt_map[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
        # 高斯模糊生成密度图
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        # 转换为张量并放大数值
        gt_map = torch.from_numpy(gt_map)# 转为Tensor
        gt_map = gt_map *60 # 放大数值（可视化需要）
        
        sample = {'image':image,'dots':dots, 'boxes':boxes, 'pos':rects, 'gt_map':gt_map, 'scale':scale}
        return sample['image'], sample['dots'], sample['boxes'], sample['pos'] ,sample['gt_map'],im_id,sample['scale']


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_test = TestData()
    print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # define the model
    model = CntViT.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)

    print(f"Start testing.")
    start_time = time.time()
    
    # test
    epoch = 0
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # 添加插值函数
    interpolate = torch.nn.functional.interpolate

    # some parameters in training
    test_mae = 0
    test_rmse = 0
    test_rmae = 0
    test_total_gt = 0
    test_total_gt_squared = 0
    pred_cnt = 0
    gt_cnt = 0
    total_flops = 0
    flops_per_sample = []

    loss_array = []
    gt_array = []
    wrong_id = []

    for data_iter_step, (samples, gt_dots, boxes, pos, gt_map,im_id,scale) in enumerate(metric_logger.log_every(data_loader_test, print_freq, header)):
        samples = samples.to(device, non_blocking=True).float()
        gt_dots = gt_dots.to(device, non_blocking=True).float()
        boxes = boxes.to(device, non_blocking=True).float()
        scale = scale.to(device, non_blocking=True).float()
        a = scale * 384
        gt_map = gt_map.to(device, non_blocking=True)

        _,_,h,w = samples.shape 

        sample_flops = 0.0  # 记录当前样本的FLOPs
        r_cnt = 0
        s_cnt = 0
        for rect in pos:
            r_cnt+=1
            if r_cnt>3:
                break
            if rect[2]-rect[0]<10 and rect[3] - rect[1]<10:
                s_cnt +=1
        
        if s_cnt >= 1:
            r_images = []
            r_images.append(TF.crop(samples[0], 0, 0, int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h/3), 0, int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h*2/3), 0, int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], 0, int(w/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h/3), int(w/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h*2/3), int(w/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], 0, int(w*2/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h/3), int(w*2/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h*2/3), int(w*2/3), int(h/3), int(w/3)))
            pred_cnt = 0
            density_maps = []
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h,w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1
                
                with torch.no_grad():
                    while start + 383 < w:
                        # 调整输入尺寸到384x384
                        patch = r_image[:,:,:,start:start+384]
                        patch = interpolate(patch, size=(384, 384), mode='bilinear', align_corners=False)
                        input_x = [patch, boxes, scale]
                        
                        # 计算当前块的FLOPs
                        with torch.no_grad():
                            macs, _ = profile(model_without_ddp, inputs=(input_x,), verbose=False)
                        sample_flops += macs
                        
                        output = model(input_x)
                        output=output.squeeze(0)

                        
                        # 将输出密度图调整回原始尺寸 (h,384)
                        output = output.unsqueeze(0).unsqueeze(0)
                        output = interpolate(output, size=(h,384), mode='bilinear', align_corners=False)
                        output = output.squeeze(0).squeeze(0)

                        # a = output.clone()
                        b1 = nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                        d1 = b1(output[:,0:prev-start+1])
                        b2 = nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                        d2 = b2(output[:,prev-start+1:384])            
                        
                        b3 = nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                        density_map_l = b3(density_map[:,0:start])
                        density_map_m = b1(density_map[:,start:prev+1])
                        b4 = nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                        density_map_r = b4(density_map[:,prev+1:w])

                        density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 +d2

                        prev = start + 383
                        start = start + 128
                        if start+383 >= w:
                            if start == w - 384 + 128: break
                            else: start = w - 384
                pred_cnt += torch.sum(density_map/60).item()
            pred_cnt_new = 0
            density_map = torch.zeros([h,w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    # 调整输入尺寸到384x384
                    patch = samples[:,:,:,start:start+384]
                    patch = interpolate(patch, size=(384, 384), mode='bilinear', align_corners=False)
                    input_x = [patch, boxes, scale]
                    
                    # 计算当前块的FLOPs
                    with torch.no_grad():
                        macs, _ = profile(model_without_ddp, inputs=(input_x,), verbose=False)
                    sample_flops += macs
                    
                    output = model(input_x)
                    output=output.squeeze(0)
                    
                    # 将输出密度图调整回原始尺寸 (h,384)
                    output = output.unsqueeze(0).unsqueeze(0)
                    output = interpolate(output, size=(h,384), mode='bilinear', align_corners=False)
                    output = output.squeeze(0).squeeze(0)

                    b1 = nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                    d1 = b1(output[:,0:prev-start+1])
                    b2 = nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                    d2 = b2(output[:,prev-start+1:384])            
                    
                    b3 = nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                    density_map_l = b3(density_map[:,0:start])
                    density_map_m = b1(density_map[:,start:prev+1])
                    b4 = nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                    density_map_r = b4(density_map[:,prev+1:w])

                    density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 +d2

                    prev = start + 383
                    start = start + 128
                    if start+383 >= w:
                        if start == w - 384 + 128: break
                        else: start = w - 384
            pred_cnt_new = torch.sum(density_map/60).item()

            e_cnt = 0
            cnt = 0
            for rect in pos:
                cnt+=1
                if cnt>3:
                    break
                e_cnt += torch.sum(density_map[rect[0]:rect[2]+1,rect[1]:rect[3]+1]/60).item()
            e_cnt = e_cnt / 3

            if e_cnt > 1.8:
                pred_cnt_new /= e_cnt
            
            if pred_cnt>pred_cnt_new * 9:
                pred_cnt = pred_cnt_new

        else: 
            density_map = torch.zeros([h,w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            
            ha = h//16
            wa = w//16
            start1 = 0
            prev1 = -1
            with torch.no_grad():
                while start + 383 < w:
                    # 调整输入尺寸到384x384
                    patch = samples[:,:,:,start:start+384]
                    patch = interpolate(patch, size=(384, 384), mode='bilinear', align_corners=False)
                    input_x = [patch, boxes, scale]
                    
                    # 计算当前块的FLOPs
                    with torch.no_grad():
                        macs, _ = profile(model_without_ddp, inputs=(input_x,), verbose=False)
                    sample_flops += macs
                    
                    output = model(input_x)
                    output=output.squeeze(0)
                    
                    # 将输出密度图调整回原始尺寸 (h,384)
                    output = output.unsqueeze(0).unsqueeze(0)
                    output = interpolate(output, size=(h,384), mode='bilinear', align_corners=False)
                    output = output.squeeze(0).squeeze(0)
                    
                    b1 = nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                    d1 = b1(output[:,0:prev-start+1])
                    b2 = nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                    d2 = b2(output[:,prev-start+1:384])            
                    
                    b3 = nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                    density_map_l = b3(density_map[:,0:start])
                    density_map_m = b1(density_map[:,start:prev+1])
                    b4 = nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                    density_map_r = b4(density_map[:,prev+1:w])

                    density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 +d2

                    prev = start + 383
                    start = start + 128
                    prev1 = start1 + 23
                    start1 = start1 + 8
                    if start+383 >= w:
                        if start == w - 384 + 128: break
                        else: start = w - 384

                    if start1+23 >= wa:
                        if start1 == wa - 23 + 8: break
                        else: start1 = wa - 23
            pred_cnt = torch.sum(density_map/60).item()

            e_cnt = 0
            cnt = 0
            for rect in pos:
                cnt+=1
                if cnt>3:
                    break
                e_cnt += torch.sum(density_map[rect[0]:rect[2]+1,rect[1]:rect[3]+1]/60).item()
            e_cnt = e_cnt / 3
            if e_cnt > 1.8:
                pred_cnt /= e_cnt
            
        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        if cnt_err > 200:
            wrong_id.append(im_id)
            print(im_id)

        test_mae += cnt_err
        test_rmse += cnt_err ** 2
        if gt_cnt != 0:
            test_rmae += cnt_err / gt_cnt
        else:
            # 当没有真实标注样本时，直接加0.0
            test_rmae += 0
        test_total_gt += gt_cnt
        test_total_gt_squared += gt_cnt**2
        
        # 记录当前样本的FLOPs
        flops_per_sample.append(sample_flops / 1e9)  # 转换为GFLOPs
        total_flops += sample_flops

        print(f'{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2}, FLOPs: {sample_flops/1e9:.2f} GFLOPs')

        loss_array.append(cnt_err)
        gt_array.append(gt_cnt)
        
        torch.cuda.synchronize(device=0)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    log_stats = {'MAE': test_mae/(len(data_loader_test)),
                'RMSE':  (test_rmse/(len(data_loader_test)))**0.5}

    n = len(data_loader_test)
    m = test_total_gt_squared - 2 * (test_total_gt/n) * test_total_gt + n * (test_total_gt/n)**2 # R2的分母
    print('MAE:{:5.2f},RMSE:{:5.2f},rMAE:{:5.2f},rRMSE:{:5.2f},R2:{:5.2f}'.format(test_mae/n,(test_rmse/n)**0.5,test_rmae/n,((test_rmse/n)**0.5)/(test_total_gt/n),1-(test_rmse/m)))
    
    # 打印FLOPs统计结果
    # ===================================================================
    print("\nFLOPs Summary:")
    print(f"Total FLOPs for all samples: {total_flops/1e12:.4f} TFLOPs")
    print(f"Average FLOPs per sample: {total_flops/len(data_loader_test)/1e9:.4f} GFLOPs")
    print(f"Min sample FLOPs: {min(flops_per_sample):.4f} GFLOPs")
    print(f"Max sample FLOPs: {max(flops_per_sample):.4f} GFLOPs")
    print(f"Median sample FLOPs: {np.median(flops_per_sample):.4f} GFLOPs")
    # ===================================================================

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)