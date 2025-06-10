# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf
import timm
import torch.nn as nn

class SumConv2d(nn.Module):
    def __init__(self):
        super(SumConv2d, self).__init__()
        kernel_size = 16
        stride = 16
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.conv.weight.data.fill_(1)
        # 禁止权重更新
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 禁止梯度计算
        x = x.unsqueeze(1)
        with torch.no_grad():
            return self.conv(x).squeeze()

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

'''
日志记录工具,用于跟踪训练过程中的各项指标(如loss、准确率等)
'''
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

'''
1.​​在分布式训练环境中控制打印行为​​（确保只有主进程或特定条件下才输出日志）
2.在输出前添加当前时间（如 [14:30:00.123456]），便于日志追踪
'''
def setup_for_distributed(is_master):
    """
    此函数在非主进程中禁用打印
    """
    builtin_print = builtins.print # 备份原始的 print 函数，以便在新 print 中调用

    # 自定义 print 函数​
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time() # 在输出前添加当前时间（如 [14:30:00.123456]），便于日志追踪
            builtin_print('[{}] '.format(now), end='')  # 打印时间戳
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

'''
分布式训练模式的初始化​​:
    ​作用​​：根据不同的分布式环境配置（如 MPI、SLURM 或环境变量），初始化 PyTorch 的分布式训练（多 GPU/多节点）。
    ​​核心任务​​：
        (1)确定当前进程的 rank(全局编号)、world_size(总进程数)、gpu(本地 GPU 编号)。
        (2)设置分布式通信的后端(如 nccl)和初始化方法(如 tcp://)。
        (3)调用 torch.distributed.init_process_group 初始化进程组
'''
def init_distributed_mode(args):
    # 一、通过检查环境变量判断当前运行环境：
    # 1.MPI 环境 (args.dist_on_itp)​
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    # 2.直接环境变量配置​
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # 3.SLURM 集群环境​
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # 4. 非分布式模式​
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    # 二、初始化分布式训练​
    args.distributed = True
    torch.cuda.set_device(args.gpu) # 绑定当前进程到指定 GPU
    args.dist_backend = 'nccl' # 使用 NCCL 后端（NVIDIA GPU 专用）
    # 打印初始化信息（仅主进程可见）
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    # 初始化进程组
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # 同步所有进程
    torch.distributed.barrier()
    # 设置分布式打印控制（仅主进程打印日志）
    setup_for_distributed(args.rank == 0)


'''
混合精度训练(AMP)
返回剪裁后的范数
'''
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()# PyTorch 提供的梯度缩放器，用于混合精度训练

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)# 缩放损失并反向传播​​
        # ​​梯度裁剪与范数计算
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # 反缩放梯度（恢复原始值）
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad) # 裁剪梯度
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters) # 计算梯度范数（未裁剪）
            self._scaler.step(optimizer)# 更新参数（自动处理缩放）
            self._scaler.update()# 调整缩放因子（根据梯度溢出情况）
        else:
            norm = None  
        return norm # 裁剪后的范数（或原始范数）

    # 保存和加载梯度缩放器的内部状态（如当前缩放因子），用于训练中断后恢复
    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
            print(f"Removing key decoder_pos_embed from pretrained checkpoint")
            del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def load_model_FSC(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # for name in checkpoint['model']:
        #     if 'block' in name:
        #         checkpoint['model']['a'] = 1
        
        # ### 我们来看一下权重的区别。
        # ckp = checkpoint['model']['patch_embed_exemplar.proj.weight']
        # ckp = torch.mean(ckp,dim=2)
        # ckp = torch.mean(ckp,dim=2)
        # ckp = torch.mean(ckp,dim=0)

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']
        
        if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
            print(f"Removing key decoder_pos_embed from pretrained checkpoint")
            del checkpoint['model']['decoder_pos_embed']
        # if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
        #     print(f"Removing key decoder_pos_embed from pretrained checkpoint")
        #     del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)

def load_model_FSC_encoder(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']
        
        # if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
        #     print(f"Removing key decoder_pos_embed from pretrained checkpoint")
        #     del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)

def load_model_FSC_one_stage(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']
        
        if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
            print(f"Removing key decoder_pos_embed from pretrained checkpoint")
            del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)

def load_model_FSC1(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            #model = timm.create_model('vit_base_patch16_224', pretrained=True)
            #torch.save(model.state_dict(), './output_abnopre_dir/checkpoint-6657.pth')
            checkpoint1 = torch.load('./output_abnopre_dir/checkpoint-6657.pth', map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        del checkpoint1['cls_token'],checkpoint1['pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_without_ddp.load_state_dict(checkpoint1, strict=False)
        print("Resume checkpoint %s" % args.resume)
        
def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def add_weight_decay_lr(model,lr_back, weight_decay=1e-5, skip_list=()):
    decay = []
    decay_backbone = []
    no_decay = []
    no_decay_backbone = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if 'blocks' in name and 'decoder' not in name:
                no_decay_backbone.append(param)
            else:
                no_decay.append(param)
        else:
            if 'blocks' in name and 'decoder' not in name: 
                decay_backbone.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': no_decay_backbone, 'weight_decay': 0.,'lr':lr_back},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': decay_backbone, 'weight_decay': weight_decay, 'lr':lr_back}]