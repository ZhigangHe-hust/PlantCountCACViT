import json
import os

def get_jpg_filenames(images_path):
    """
    获取目录下所有图片的文件名
    :param root_dir: 根目录的路径
    :return: 包含所有图片文件名的列表
    """
    jpg_filenames = []
    for filename in os.listdir(images_path):
        if filename.lower().endswith('.jpg'):
            filename_without_ext = os.path.splitext(filename)[0]
            jpg_filenames.append(filename_without_ext)
    
    return jpg_filenames

if __name__ == "__main__":

    images_path ='./data/images'
    jpg_filenames = get_jpg_filenames(images_path)
    jpg_fileclasses = []

    for filename in jpg_filenames:
        last_underscore = filename.rfind('_')  # 找到最后一个 _ 的位置
        jpg_fileclasses.append(filename[:last_underscore])
    
    jpg_filenames_with_ext = [filename + ".jpg" for filename in jpg_filenames]

    # # 检验读取情况
    # print(f"The number of jpg_filenames_with_ext {len(jpg_filenames_with_ext)} ")
    # print(f"The number of jpg_fileclasses {len(jpg_fileclasses)} ")
    # for i, name in enumerate(jpg_filenames_with_ext[:5]): # 打印列表前5个
    #     print(f"jpg_filenames_with_ext:\n {i+1}. {name}")
    # for i, name in enumerate(jpg_fileclasses[:5]): 
    #     print(f"jpg_fileclasses:\n {i+1}. {name}")

    # 确保两个列表长度相同
    if len(jpg_filenames_with_ext) != len(jpg_fileclasses):
        print("Error: The lengths of the two lists are inconsistent!")
        exit(1)

    # 写入文本文件
    output_file = "./data/ImageClasses.txt"

    with open(output_file, 'w') as f:
        for img, cls in zip(jpg_filenames_with_ext, jpg_fileclasses):
            # 每行的格式：图片名称 + 空格 + 类别
            line = f"{img} {cls}\n"
            f.write(line)

    print(f"Successfully wrote {len(jpg_filenames_with_ext)} data to {output_file}.")

    # # 验证写入内容
    # with open(output_file, 'r') as f:
    #     content = f.read()
    #     print("File Contents:")
    #     print(content)



