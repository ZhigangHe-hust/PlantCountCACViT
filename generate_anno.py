import glob
import os
import os
import pandas as pd
import json

def generate_annotation_entry(regions, filename):
    box_coords = []
    points = []

    for region in regions:
        shape_attrs = json.loads(region['region_shape_attributes'])

        if shape_attrs.get('name') == 'rect':
            # 处理矩形框 rect 类型
            x = shape_attrs['x']
            y = shape_attrs['y']
            width = shape_attrs['width']
            height = shape_attrs['height']

            # 四个角点顺序：左上、右上、右下、左下
            top_left = [x, y]
            top_right = [x + width, y]
            bottom_right = [x + width, y + height]
            bottom_left = [x, y + height]

            box_coords.append([top_left, top_right, bottom_right, bottom_left])

        elif shape_attrs.get('name') == 'point':
            # 处理点类型
            cx = shape_attrs.get('cx')
            cy = shape_attrs.get('cy')
            points.append([cx, cy])

    result = {}
    if box_coords:
        result["box_examples_coordinates"] = box_coords
    if points:
        result["points"] = points

    return {filename: result}


def find_matching_rows(a_str, root_dir):
    # 1. 提取文件夹B名称：第二个 '_' 前的部分
    parts = a_str.split('_', 2)  # 只分割前两个
    folder_b = '_'.join(parts[:2])  # Vatica_mangachapoi

    # 2. 提取文件夹C名称 & CSV文件名：最后一个 '_' 前的部分
    base_name = '_'.join(a_str.split('_')[:-1])  # Vatica_mangachapoi_yellow_fruit

    # 3. 构建路径
    path_b = os.path.join(root_dir, folder_b)
    path_c = os.path.join(path_b, base_name)

    csv_file = os.path.join(path_c, 'labels', f"{base_name}.csv")

    if not os.path.isfile(csv_file):
        print(f"CSV 文件不存在: {csv_file}")
        return []

    # 5. 读取 CSV 文件
    try:
        df = pd.read_csv(csv_file, header=0)
    except Exception as e:
        print(f"读取 CSV 文件失败: {csv_file}, 错误: {e}")
        return []

    if df.empty:
        print(f"CSV 文件为空: {csv_file}")
        return []

    matches = df.iloc[:, 0].astype(str).str.startswith(a_str + '.')

    if matches.sum() == 0:
        print(f"没有匹配到任何行: {a_str}")
    result = df[matches]
    subset = result.iloc[:, [5]]  # 第6列是索引5
    return subset.to_dict(orient='records')


def get_jpg_filenames(directory):
    filenames = []
    # 使用 glob 匹配 .jpg 文件（包括子目录中的文件）
    jpg_files = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True)
    for file_path in jpg_files:
        # 获取文件名并去掉 .jpg 扩展名
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        filenames.append(name_without_ext)
    return filenames

# 使用方法
directory_path = './data/images'  # 图片文件夹路径
label_path = 'Datasets'  # 数据集文件夹路径
jpg_filenames_list = get_jpg_filenames(directory_path)
final_output = {}
for jpg_filename in jpg_filenames_list:
    matching_rows = find_matching_rows(jpg_filename, label_path)
    if matching_rows:
        # print(f"匹配到的行: {matching_rows}")
        result_dict = generate_annotation_entry(matching_rows, jpg_filename)
        final_output.update(result_dict)
    else:
        print(f"没有匹配到任何行: {jpg_filename}")

print(f"The number of images {len(jpg_filenames_list)} ")
# print(jpg_filenames_list)

# with open("our_data/annotations.json", "w") as f:
#     json.dump(final_output, f, indent=4)
