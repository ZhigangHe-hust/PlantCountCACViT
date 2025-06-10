import json
import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

data_path = './data/'
im_dir = data_path + 'images'
anno_file = data_path + 'annotations.json'
data_split_file = data_path + 'Train_Val_Test.json'
class_file = data_path + 'ImageClasses.txt'

### 看下CounTR的数据怎么处理的

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
train_set = data_split['train']

# -----------------------------------------------------------------------------------------------------------------------------

"""
resizePreTrainImage:
调整图片大小，使：
    1. 图片尺寸为 384 * 384
    2. 新的高度和宽度能被 16 整除
    3. 保留宽高比
密度和边框的正确性不保留（裁剪和水平翻转）
"""
class resizePreTrainImage(object):
    
    def __init__(self, MAX_HW=384):
        self.max_hw = MAX_HW
        self.scale_number = 20

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        # 将图片变成可以被16整除的最近的值，然后进行resize。
        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        '''scale_factor = float(256)/ H
        new_H = 16*int(H*scale_factor/16)
        new_W = 16*int(W*scale_factor/16)'''
        # 通过cv的resize函数对density进行resize。
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))
        # 通过两步来完成density的重新选择
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)
        
        if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        boxes = list()
        ## 获取尺度信息
        scale_embedding = []
        for box in lines_boxes:
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            # boxes.append([0, y1,x1,y2,x2])
            bbox = resized_image.crop((x1,y1,x2,y2))
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(TTensor(bbox))
        
            scale = (x2 - x1) / new_W * 0.5 + (y2 -y1) / new_H * 0.5 
            # scale = scale // (0.5 / self.scale_number)    ## 大概的意义就是，一共分为20个尺度等级，从0-19（设置20），每个尺度差距为1/40 img size。
            # scale = scale if scale < self.scale_number - 1 else self.scale_number - 1 ##
            scale_embedding.append(torch.tensor(scale))
        scale_embedding = scale_embedding
        scale_embedding = torch.stack(scale_embedding, dim=0)
        boxes = torch.stack(boxes,dim=0)
        resized_image = PreTrainNormalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density,'scale':scale_embedding}
        return sample

"""
resizeTrainImage:
调整图像大小，以便：
    1. 图像等于 384 * 384
    2. 新的高度和宽度可以被 16 整除
    3. 宽高比可能保持不变
密度图被裁剪，使其与裁剪后的图像具有相同的大小（和位置）。
示例框可能位于裁剪区域之外。
增强算法包括高斯噪声、色彩抖动、高斯模糊、随机仿射、随机水平翻转和马赛克（如果没有马赛克，则使用随机裁剪）。
"""
class resizeTrainImage(object):

    def __init__(self, MAX_HW=384):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        # 首先获得图片的信息，包括密度图，标记等等
        image, lines_boxes, density, dots, im_id, m_flag = sample['image'], sample['lines_boxes'],sample['gt_density'], sample['dots'], sample['id'], sample['m_flag']
        # 获得尺寸
        W, H = image.size
        # 进行resize，使得可以被16刚刚好整除。
        new_H = max(16 * (H // 16), 384)
        new_W = max(16 * (W // 16), 384)
        scale_factor = float(new_W)/ W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))   
        
        # Augmentation probability 设置数据增强的概率
        aug_p = random.random()
        aug_flag = 0
        mosaic_flag = 0
        if aug_p < 0.4: # 0.4
            aug_flag = 1
            if aug_p < 0.25: # 0.25
                aug_flag = 0
                mosaic_flag = 1

        # Gaussian noise 高斯噪声
        resized_image = TTensor(resized_image)
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

        # Color jitter and Gaussian blur 颜色的翻转，和高斯滤波
        if aug_flag == 1:
            re_image = Augmentation(re_image)

        # Random affine 随机仿射变换
        if aug_flag == 1:
            re1_image = re_image.transpose(0,1).transpose(1,2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(Keypoint(x=min(new_W-1,int(dots[i][0]*scale_factor)), y=min(new_H-1,int(dots[i][1]))))
            kps = KeypointsOnImage(keypoints, re1_image.shape)

            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-15,15),
                    scale=(0.8, 1.2),
                    shear=(-10,10),
                    translate_percent={"x": (-0.2,0.2), "y": (-0.2,0.2)}
                )
            ])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Produce dot annotation map 制作点注记图
            resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
            for i in range(len(kps.keypoints)):
                if(int(kps_aug.keypoints[i].y)<= new_H-1 and int(kps_aug.keypoints[i].x)<=new_W-1) and not kps_aug.keypoints[i].is_out_of_image(re1_image):
                    resized_density[int(kps_aug.keypoints[i].y)][int(kps_aug.keypoints[i].x)]=1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

        # Random horizontal flip 随机水平翻转
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)
        
        # Random 384*384 crop in a new_W*384 image and 384*new_W density map 在图片中随机裁剪
        if mosaic_flag == 0:
            if aug_flag == 0:
                re_image = resized_image
                resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
                for i in range(dots.shape[0]):
                    resized_density[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
                resized_density = torch.from_numpy(resized_density)
            # 随机裁剪
            start = random.randint(0, new_W-1-383)
            reresized_image = TF.crop(re_image, 0, start, 384, 384)
            # reresized_density = resized_density[:, start:start+384]
            reresized_density = resized_density[:384, start:start+384] # # 修改为（同时裁剪高度和宽度）
        
        # Random self mosaic 随机自马赛克化
        else:
            image_array = []
            map_array = []
            blending_l = random.randint(10, 20)
            resize_l = 192 + 2 * blending_l
            if dots.shape[0] >= 70:
                for i in range(4):
                    length =  random.randint(150, 384)
                    start_W = random.randint(0, new_W-length)
                    start_H = random.randint(0, new_H-length)
                    reresized_image1 = TF.crop(resized_image, start_H, start_W, length, length)
                    reresized_image1 = transforms.Resize((resize_l, resize_l))(reresized_image1)
                    reresized_density1 = np.zeros((resize_l,resize_l),dtype='float32')
                    for i in range(dots.shape[0]):
                        if min(new_H-1,int(dots[i][1])) >= start_H and min(new_H-1,int(dots[i][1])) < start_H + length and min(new_W-1,int(dots[i][0]*scale_factor)) >= start_W and min(new_W-1,int(dots[i][0]*scale_factor)) < start_W + length:
                            reresized_density1[min(resize_l-1,int((min(new_H-1,int(dots[i][1]))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_W-1,int(dots[i][0]*scale_factor))-start_W)*resize_l/length))]=1
                    reresized_density1 = torch.from_numpy(reresized_density1)
                    image_array.append(reresized_image1)
                    map_array.append(reresized_density1)
            else:
                m_flag = 1
                prob = random.random()
                if prob > 0.25:
                    gt_pos = random.randint(0,3)
                else:
                    gt_pos = random.randint(0,4) # 5% 0 objects

                # 修复点1: 添加对辅助图像尺寸的验证
                MIN_MOSAIC_SIZE = 100  # 马赛克增强所需的最小图像尺寸

                for i in range(4):
                    if i == gt_pos:
                        Tim_id = im_id
                        r_image = resized_image
                        Tdots = dots
                        new_TH = new_H
                        new_TW = new_W
                        Tscale_factor = scale_factor
                    else:
                        Tim_id = train_set[random.randint(0, len(train_set)-1)]
                        Tdots = np.array(annotations[Tim_id]['points'])
                        '''while(abs(Tdots.shape[0]-dots.shape[0]<=10)):
                            Tim_id = train_set[random.randint(0, len(train_set)-1)]
                            Tdots = np.array(annotations[Tim_id]['points'])'''
                        Timage = Image.open('{}/{}'.format(im_dir, Tim_id))
                        Timage.load()
                        # 修复点2: 确保图像最小尺寸
                        new_TH = max(16 * (Timage.size[1] // 16), MIN_MOSAIC_SIZE)
                        new_TW = max(16 * (Timage.size[0] // 16), MIN_MOSAIC_SIZE)
                        Tscale_factor = float(new_TW)/ Timage.size[0]
                        r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))

                        # 修复点3: 安全的裁剪长度计算
                    max_length = min(384, min(new_TH, new_TW))
                    
                    if max_length < 250:
                        # 图像太小，只能使用实际最大长度
                        length = max_length
                        start_W = 0
                        start_H = 0
                    else:
                        length = random.randint(250, max_length)
                        
                        # 修复点4: 安全的随机位置生成
                        def safe_randint(min_val, max_val):
                            """安全的随机整数生成，处理无效范围"""
                            if max_val < min_val:
                                return min_val
                            return random.randint(min_val, max_val)
                        
                        start_W = safe_randint(0, max(0, new_TW - length))
                        start_H = safe_randint(0, max(0, new_TH - length))

                    # 修复点5: 安全的裁剪操作
                    r_image1 = TF.crop(r_image, 
                                      min(start_H, new_TH - 1), 
                                      min(start_W, new_TW - 1), 
                                      min(length, new_TH - start_H), 
                                      min(length, new_TW - start_W))
                    r_image1 = transforms.Resize((resize_l, resize_l))(r_image1)
                    
                    # 清理文件名，确保类别比较正确
                    clean_im_id = im_id.split('/')[-1].split('.')[0].strip()
                    clean_Tim_id = Tim_id.split('/')[-1].split('.')[0].strip()
                    
                    r_density1 = np.zeros((resize_l, resize_l), dtype='float32')
                    if class_dict.get(clean_im_id) == class_dict.get(clean_Tim_id):
                        for j in range(Tdots.shape[0]):  # 改为j避免与外部循环冲突
                            # 添加边界检查
                            y = min(new_TH-1, int(Tdots[j][1]))
                            x = min(new_TW-1, int(Tdots[j][0]*Tscale_factor))
                            
                            if (y >= start_H and y < start_H + length and
                                x >= start_W and x < start_W + length):
                                target_y = min(resize_l-1, int((y - start_H) * resize_l / length))
                                target_x = min(resize_l-1, int((x - start_W) * resize_l / length))
                                r_density1[target_y][target_x] = 1
                    
                    r_density1 = torch.from_numpy(r_density1)
                    image_array.append(r_image1)
                    map_array.append(r_density1)

            reresized_image5 = torch.cat((image_array[0][:,blending_l:resize_l-blending_l],image_array[1][:,blending_l:resize_l-blending_l]),1)
            reresized_density5 = torch.cat((map_array[0][blending_l:resize_l-blending_l],map_array[1][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image5[:,192+i] = image_array[0][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image5[:,191-i] = image_array[1][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image5 = torch.clamp(reresized_image5, 0, 1)

            reresized_image6 = torch.cat((image_array[2][:,blending_l:resize_l-blending_l],image_array[3][:,blending_l:resize_l-blending_l]),1)
            reresized_density6 = torch.cat((map_array[2][blending_l:resize_l-blending_l],map_array[3][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image6[:,192+i] = image_array[2][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image6[:,191-i] = image_array[3][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image6 = torch.clamp(reresized_image6, 0, 1)

            reresized_image = torch.cat((reresized_image5[:,:,blending_l:resize_l-blending_l],reresized_image6[:,:,blending_l:resize_l-blending_l]),2)
            reresized_density = torch.cat((reresized_density5[:,blending_l:resize_l-blending_l],reresized_density6[:,blending_l:resize_l-blending_l]),1)
            for i in range(blending_l):
                    reresized_image[:,:,192+i] = reresized_image5[:,:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image[:,:,191-i] = reresized_image6[:,:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image = torch.clamp(reresized_image, 0, 1)
        
        # Gaussian distribution density map 高斯分布密度图
        reresized_density = ndimage.gaussian_filter(reresized_density.numpy(), sigma=(1, 1), order=0)

        # Density map scale up 密度图放大 
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density)
        
        # Crop bboxes and resize as 64x64 裁剪框并调整大小为 64x64
        boxes = list()
        scale_x = []
        scale_y = []
        cnt = 0
        for box in lines_boxes:
            cnt+=1
            if cnt>3:
                break
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], int(box2[1]*scale_factor), box2[2], int(box2[3]*scale_factor)
            scale_x1 = torch.tensor((x2-x1+1))/384
            scale_y1 = torch.tensor((y2-y1+1))/384
            scale_x.append(scale_x1)
            scale_y.append(scale_y1)
            bbox = resized_image[:,y1:y2+1,x1:x2+1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
        scale_xx = torch.stack(scale_x).unsqueeze(-1)
        scale_yy = torch.stack(scale_y).unsqueeze(-1)
        scale = torch.cat((scale_xx,scale_yy),dim=1)

        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)
        
        # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]       
        sample = {'image':reresized_image,'boxes':boxes,'gt_density':reresized_density, 'm_flag': m_flag, 'scale':scale}

        return sample

# -----------------------------------------------------------------------------------------------------------------------------



"""
预训练时的数据增强(随机裁剪+翻转+转张量):
    1.RandomResizedCrop
    随机裁剪并缩放到 384x384 大小。
    scale=(0.2, 1.0)：裁剪区域占原图面积的 20%~100%。
    interpolation=3:使用双三次插值(InterpolationMode.BICUBIC)
    2.RandomHorizontalFlip
    以 50% 概率水平翻转图像（增加数据多样性）。
    3.ToTensor
    将 PIL 图像或 NumPy 数组转为 PyTorch 张量（形状 CxHxW,值范围 [0, 1])
    4.注释掉的归一化
    若取消注释，会按 IM_NORM_MEAN 和 IM_NORM_STD 标准化张量。
"""
PreTrainNormalize = transforms.Compose([   
        transforms.RandomResizedCrop(384, scale=(0.2, 1.0), interpolation=3), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

"""
仅将图像转为张量（无增强或归一化）
"""
TTensor = transforms.Compose([   
        transforms.ToTensor(),
        ])

"""
额外的数据增强:
    1.ColorJitter:
    随机调整图像的亮度(brightness)、对比度(contrast)、饱和度(saturation)和色调(hue)。
    参数范围:0.25 表示亮度在 [0.75, 1.25] 之间随机变化（其他类似）。
    2.GaussianBlur
    高斯模糊（模糊核大小 7x7 到 9x9 随机选择）。
    注意:kernel_size 必须是奇数
"""
Augmentation = transforms.Compose([   
        transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
        transforms.GaussianBlur(kernel_size=(7,9))
        ])


"""
标准化图像（常用于验证/测试阶段）:
    1.ToTensor
    转为张量并缩放到 [0, 1]
    2.Normalize
    按均值 IM_NORM_MEAN 和标准差 IM_NORM_STD 归一化
    公式:output = (input - mean) / std。
    假设 IM_NORM_MEAN = [0.485, 0.456, 0.406](ImageNet均值),std = [0.229, 0.224, 0.225]
"""
Normalize = transforms.Compose([   
        transforms.ToTensor(),
        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

# -----------------------------------------------------------------------------------------------------------------------------

# 对训练数据进行处理
TransformTrain = transforms.Compose([resizeTrainImage(MAX_HW)])

# 对预训练数据进行处理
TransformPreTrain = transforms.Compose([resizePreTrainImage(MAX_HW)])

