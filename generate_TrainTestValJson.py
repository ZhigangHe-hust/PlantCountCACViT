import json
import os

def get_jpg_filenames(root_dir):
    """
    获取根目录下所有图片(包括png和jpg)的文件名
    :param root_dir: 根目录的路径
    :return: 包含所有图片文件名的列表
    """
    filenames = []
    
    # 遍历第一层：植物种类文件夹
    for plant_dir in os.listdir(root_dir):
        plant_path = os.path.join(root_dir, plant_dir)
        
        if not os.path.isdir(plant_path):
            continue
            
        # 遍历第二层：状态文件夹
        for state_dir in os.listdir(plant_path):
            state_path = os.path.join(plant_path, state_dir)
            
            if not os.path.isdir(state_path):
                continue
                
            # 进入images子文件夹
            images_path = os.path.join(state_path, 'images')
            if os.path.exists(images_path) and os.path.isdir(images_path):
                # 获取所有图片文件名
                for filename in os.listdir(images_path):
                    if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
                        filename_without_ext = os.path.splitext(filename)[0]
                        filenames.append(filename_without_ext)

            # 统一后缀为jpg
            jpg_filenames = [name + ".jpg" for name in filenames]                    
    
    return jpg_filenames


if __name__ == "__main__":

    train_filename_list = [] # 存储训练集文件名的列表
    test_filename_list = []  # 存储测试集文件名的列表
    val_filename_list = []   # 存储验证集文件名的列表

    traindata_dir = "./data/dataset/Train" # 训练集的绝对路径(改为你的绝对路径)
    testdata_dir = "./data/dataset/Test"   # 测试集的绝对路径
    valdata_dir = "./data/dataset/Val"     # 验证集的绝对路径

    
    # 获取所有图片文件
    train_filename_list = get_jpg_filenames(traindata_dir)
    test_filename_list = get_jpg_filenames(testdata_dir)
    val_filename_list = get_jpg_filenames(valdata_dir)

    # 检验读取情况
    print(f"The number of train images {len(train_filename_list)} ")
    print(f"The number of test images {len(test_filename_list)} ")
    print(f"The number of val images {len(val_filename_list)} ")
    for i, name in enumerate(train_filename_list[:5]): # 打印列表前5个
        print(f"train_filename_list:\n {i+1}. {name}")
    for i, name in enumerate(test_filename_list[:5]): 
        print(f"test_filename_list:\n {i+1}. {name}")
    for i, name in enumerate(val_filename_list[:5]): 
        print(f"val_filename_list:\n {i+1}. {name}")

    # 结果字典
    result_dict = {
    "train": train_filename_list,
    "test": test_filename_list,
    "val": val_filename_list
    }

    # 将结果写入JSON文件
    output_file = './data/Train_Val_Test.json'

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 使用indent参数使JSON格式化输出，便于阅读
            json.dump(result_dict, f, indent=4, ensure_ascii=False)
        print(f"Success: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
