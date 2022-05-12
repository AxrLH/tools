import os
import cv2
import json
import argparse
from tqdm import tqdm


# TODO
"""
root path 
    classes.txt
    images
        jpg
    labels
        txt

coco path
    save json 
"""
parser = argparse.ArgumentParser('Datasets converter from YOLO to COCO')
parser.add_argument('--root_path', type=str, default=None,
                    help='root path of yolo images labels and classes.txt')
parser.add_argument('--coco_path', type=str, default=None,
                    help='root path of your coco datasets')
parser.add_argument('--save_path', type=str, default='instances_train2017.json',
                    help='train or val json save path')

args = parser.parse_args()


def get_train_and_test_txt_file(img_paths):
    # 根图片生成train.txt， val.txt等包含图片路径的文件,这个函数可用可不用，留作备用
    parent = os.path.abspath(os.path.join(img_paths, os.path.pardir))
    file = open(parent + 'img_split.txt', 'w+')
    for img in os.listdir(img_paths):
        file.write('\n'.join(os.path.join(img_paths, img)))
    file.close()


def yolo2coco(arg):
    root_path = arg.root_path
    print(f'loading data from {root_path}')

    assert os.path.exists(root_path)

    # 注意转换前label和img的文件名要和下面统一
    yolo_label_path = os.path.join(root_path, 'labels')
    yolo_image_path = os.path.join(root_path, 'images')

    # 打开classes.txt，没有的话自己去创一个，安装yolo里class的循序
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
    # image dir
    image_indexs = os.listdir(yolo_image_path)

    # 创建json文件
    dataset = {'categories': [], 'annotations': [], 'images': []}

    # json写入categories关键字信息,id从1开始，因为coco是这样的
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': i - 1})

    # 写入json的annotations的id, coco的id都是从1开始,imgaes和 annotation是都是
    ann_id = 1
    for j, item in enumerate(tqdm(image_indexs), start=1):
        # 支持jpg，png，其他格式自己尝试加上
        img_to_ann = item.replace('.jpg', '.txt').replace('.png', '.txt')

        # 读取w，h
        img_size = cv2.imread(os.path.join(yolo_image_path, item))
        # TODO:记得修改：img_size.shape[0:2],默认flir v2（512,640）
        height, width = img_size.shape[0:2]
        # 添加json的images关键字
        dataset['images'].append({
            'file_name': item,
            'id': j,
            'height': height,
            'width': width
        })
        # 如果没有labels信息，就只保留图像，跳过这张图的annotations关键字信息，说明这张图没有物体
        if not os.path.exists(os.path.join(yolo_label_path, img_to_ann)):
            continue

        # 填写json的annotations信息
        with open(os.path.join(yolo_label_path, img_to_ann), 'r') as f_txt:
            label_list = f_txt.readlines()
            for label in label_list:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # 照着coco2yolo逆转换
                x1 = (x - w / 2) * width
                y1 = (y - h / 2) * height
                x2 = (x + w / 2) * width
                y2 = (y + h / 2) * height

                # coco的id都是从1开始
                class_id = int(label[0]) + 1
                coco_w = max(0, x2 - x1)
                coco_h = max(0, y2 - y1)

                dataset['annotations'].append({
                    'area': coco_h * coco_w,
                    'bbox': [x1, y1, coco_w, coco_h],
                    'category_id': class_id,
                    'id': ann_id,
                    'image_id': j,
                    'iscrowd': 0,
                    # 分割是矩阵从左上角顺时针四个点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id += 1

    # 保存json
    json_dir = os.path.join(arg.coco_path, 'annotations')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    save_path = arg.save_path
    json_name = os.path.join(json_dir, save_path)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
        print(f'{arg.save_path} has been saved in {json_name}')


if __name__ == '__main__':
    # 如果要分割训练集和测试集就执行下面命令，否则跳过
    # flie v2分割好了训练和验证集，我直接跳过
    # get_train_and_test_txt_file()

    # yolo2coco,包括yolov5
    yolo2coco(args)
