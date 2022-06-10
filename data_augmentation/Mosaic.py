import argparse
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# TODO:将yolo格式的数据集通过Mosaic data augmentation，之后在转换为yolo格式的 label，img保存
"""
MOSAIC格式
[picture1_path bbox1 bbox2...
 picture2_path bbox1 bbox2...
 picture3_path bbox1 bbox2...
 picture4_path bbox1 bbox2...
]
bbox格式
x_min,y_min,x_max,y_max,cls_num

_____________________________________

yolo格式:.txt
[
cls_id x y w h 
]

_____________________________________
yolo -> mosaic -> yolo
"""
parser = argparse.ArgumentParser('MOSAIC dataset augmentation')
parser.add_argument('--yolo_path', type=str, default=None,
                    help='yolo path include images and labels document')
parser.add_argument('--mosaic_path', type=str, default=None,
                    help='mosaic path include labels document')
parser.add_argument('--img_path', type=str, default=None,
                    help='img path include after mosaic images')
parser.add_argument('--label_path', type=str, default=None,
                    help='label path include after mosaic labels')
parser.add_argument('--input_size', type=tuple, default=(1080, 1080),
                    help='input size to images')
args = parser.parse_args()


def yolo2mosaic(root_path, save_path):
    assert os.path.exists(root_path)
    # 注意转换前label和img的文件名要和下面统一
    # TODO：这里稍微修改，根据自己数据集格式
    # train_or_val = str(arg.file)
    yolo_label_path = os.path.join(os.path.join(root_path, 'labels'))
    yolo_image_path = os.path.join(os.path.join(root_path, 'images'))

    image_index = os.listdir(yolo_image_path)

    # image item from image path
    temp_img = []
    file = open(save_path + "/" + str(1) + ".txt", "w")
    for j, item in enumerate(tqdm(image_index), start=1):
        # img and annotation, support png and jpg
        img2ann = item.replace('.jpg', '.txt').replace('.png', '.txt')

        image_path = os.path.join(yolo_image_path, item)
        image_item = cv2.imread(image_path)
        height, width = image_item[0:2]

        with open(os.path.join(yolo_label_path, img2ann), 'r') as f:
            label_list = f.readlines()

            file.write(image_path)
            file.write(' ')
            for label in label_list:
                label = label.strip().split(' ')
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])
                x_center = x * width
                y_center = y * height
                x_min = x_center - (w * width / 2)
                x_max = x_center + (w * width / 2)
                y_min = y_center - (h * height / 2)
                y_max = y_center + (h * height / 2)
                cls_id = int(label[0])
                temp_img.append((x_min[1][1], y_min[1][1], x_max[1][1], y_max[1][1], cls_id))
                # temp_img.append([int(x_min[1][1]), int(y_min[1][1]), int(x_max[1][1]), int(y_max[1][1]), cls_id])
                file.write(' '.join('%d,%d,%d,%d,%d' % data for data in temp_img))
                temp_img = []

                file.write(' ')
        file.write('\n')
        if j % 4 == 0:
            file.close()
            if j != len(image_index):
                file = open(save_path + "/" + str(j // 4 + 1) + ".txt", "w")
    print('yolo label to mosaic label is done!')


def mosaic2yolo(image_data, label_data, id, save_img, save_label, input_size=args.input_size):
    # label data = [cls_id x_min y_min x_max y_max]
    img = Image.fromarray(image_data.astype(np.uint8))
    img_name = "mosaic_" + str(id)
    img.save(os.path.join(save_img, img_name) + ".jpg")
    file_label = open(save_label + "/" + img_name + ".txt", "w")
    width, height = input_size

    temp_ann = []

    for i in range(len(label_data)):
        for box in label_data:
            cls_id, x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3], box[4]
            cls_id = int(cls_id)
            x_center, y_center = (x_max - x_min) / 2, (y_max - y_min) / 2
            w_box, h_box = (x_max - x_min), (y_max - y_min)

            # 归一化
            x, y = (x_center / width), (y_center / height)
            w, h = (w_box / width), (h_box / height)

            temp_ann.append(
                (cls_id, x, y, w, h)
            )

    file_label.write('\n'.join('%d %.3f %.3f %.3f %.3f' % data for data in temp_ann))
    file_label.close()


def random_data(a=1.0, b=0.):
    return np.random.rand() * (a - b) + b


def process_bbox(old_bboxes, cutx, cuty, w, h):
    merge_bboxes = []
    # TODO:
    """
    old bboxes = [pic1 pic2 pic3 pic4]
    pic i = [x_min y_min x_max y_max cls_num x_min y_min x_max y_max cls_num ...]
    """
    for pic in range(len(old_bboxes)):
        for bbox in old_bboxes[pic]:
            temp_bbox = []

            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

            if pic == 0:
                if x_min > cutx or y_min > cuty:
                    continue
                if 0 < x_min <= cutx <= x_max:
                    x_max = cutx
                if x_min <= 0 and x_max >= cutx:
                    x_min = 0
                    x_max = cutx
                if 0 < y_min <= cuty <= y_max:
                    y_max = cuty
                if y_min <= 0 and y_max >= cuty:
                    y_min = 0
                    y_max = cuty
            if pic == 1:
                if x_min > cutx or y_max < cuty:
                    continue
                if 0 < x_min <= cutx <= x_max:
                    x_max = cutx
                if x_min <= 0 and x_max >= cutx:
                    x_min = 0
                    x_max = cutx
                if cuty < y_min < h < y_max:
                    y_max = h
                if y_min <= cuty and y_max >= h:
                    y_min = cuty
                    y_max = h
            if pic == 2:
                if x_max < cutx or y_max < cuty:
                    continue
                if cutx < x_min < w < x_max:
                    x_max = cutx
                if x_min <= cutx and x_max >= w:
                    x_min = cutx
                    x_max = w
                if cuty < y_min < h < y_max:
                    y_max = h
                if y_min <= cuty and y_max >= h:
                    y_min = cuty
                    y_max = h
            if pic == 3:
                if x_max < cutx or y_min > cuty:
                    continue
                if cutx < x_min < w < x_max:
                    x_max = cutx
                if x_min <= cutx and x_max >= w:
                    x_min = cutx
                    x_max = w
                if 0 < y_min <= cuty <= y_max:
                    y_max = cuty
                if y_min <= 0 and y_max >= cuty:
                    y_min = 0
                    y_max = cuty
            temp_bbox.append(bbox[-1])
            temp_bbox.append(x_min)
            temp_bbox.append(y_min)
            temp_bbox.append(x_max)
            temp_bbox.append(y_max)
            merge_bboxes.append(temp_bbox)
    merge_bboxes = np.array(merge_bboxes)

    return merge_bboxes


def mosaic(annotations, min_offset=0.4, input_size=args.input_size):
    # input size = (w, h)
    w, h = input_size
    min_offset_x = min_offset
    min_offset_y = min_offset
    cutx = w * min_offset_x
    cuty = h * min_offset_y

    image_datas = []
    bbox_datas = []

    index = 0

    # annotations :[image_path1 bbox1 bbox2 ... image_path4 bbox1 bbox2 ...]
    # bbox1 = x_min,y_min,x_max,y_max,cls_num
    for ann in annotations:
        # split image's message
        ann_split = ann.strip().split(' ')
        # image path
        image = Image.open(ann_split[0])
        image = image.convert('RGB')
        # IMAGE size
        iw, ih = image.size

        temp = np.zeros((len(ann_split[1:]), 5))

        # split bbox's message
        i = 0
        for bb in ann_split[1:]:
            for j in range(5):
                temp[i][j] = bb.split(',')[j]
                if j == 4:
                    i = i + 1

        box = temp
        # print(f' box {box}')

        # zoom image
        ratio = iw / ih * random_data(1.3, .7) / random_data(1.3, .7)
        scale = random_data(1, .4)
        if ratio < 1:
            nh = int(h * scale)
            nw = int(nh * ratio)
        else:
            nw = int(w * scale)
            nh = int(nw / ratio)
        # TODO:图像铺满new image (决定还是不要铺满了，铺满会拉伸图形变形，维持原始比例缩放比较好)
        # point = 1.1
        # while True:
        #     if cutx > nw or cuty > nh or (w - cutx) > nw or (h - cuty) > nh:
        #         nw = nw * point
        #         nh = nh * point
        #         point = point + .1
        #     else:
        #         break
        nw, nh = int(nw), int(nh)

        # resize each image
        image = image.resize((nw, nh), Image.BICUBIC)

        # image position :
        """
        ------------------------
        |  pic1      |  pic4   |
        |            |         |
        ------------------------
        |  pic2      |  pic3   |
        |            |         |
        ------------------------
        """
        dx, dy = 0, 0
        d_x, d_y = int(cutx) - int(nw), int(cuty) - int(nh)
        if index == 0:
            dx = d_x if d_x > 0 else 0
            dy = d_y if d_y > 0 else 0
        elif index == 1:
            dx = d_x if d_x > 0 else 0
            dy = int(cuty)
        elif index == 2:
            dx = int(cutx)
            dy = int(cuty)
        elif index == 3:
            dx = int(cutx)
            dy = d_y if d_y > 0 else 0

        new_image = Image.new('RGB', (w, h), (255, 255, 255))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)

        index = index + 1

        # resize bbox
        bbox_data = np.zeros((len(box), 5))
        if len(box) > 0:
            for i in range(len(box)):
                box[i][0] = box[i][0] * nw / iw + dx
                box[i][2] = box[i][2] * nw / iw + dx
                box[i][1] = box[i][1] * nh / ih + dy
                box[i][3] = box[i][3] * nh / ih + dy
                if box[i][0] < 0:
                    box[i][0] = 0
                if box[i][1] < 0:
                    box[i][1] = 0
                if box[i][2] > w:
                    box[i][2] = w
                if box[i][3] > h:
                    box[i][3] = h
                bbox_w = box[i][2] - box[i][0]
                bbox_h = box[i][3] - box[i][1]
                bbox_data[i] = box[i]

        image_datas.append(image_data)
        bbox_datas.append(bbox_data)

    # merge image to new image
    cutx, cuty = int(cutx), int(cuty)
    out_image = np.zeros([w, h, 3])
    out_image[:cutx, :cuty, :] = image_datas[0][:cutx, :cuty, :]
    out_image[cutx:, :cuty, :] = image_datas[1][cutx:, :cuty, :]
    out_image[cutx:, cuty:, :] = image_datas[2][cutx:, cuty:, :]
    out_image[:cutx, cuty:, :] = image_datas[3][:cutx, cuty:, :]

    out_image = np.array(out_image)

    # mosaic内容已完成，后面还可以在做一些别的变换, e.g. HSV

    # out bbox = [cls_num x_min y_min x_max y_max]
    out_bbox = process_bbox(bbox_datas, cutx, cuty, w, h)

    return out_image, out_bbox


if __name__ == '__main__':
    """
    modus:
    yolo2mosaic ; mosaic2yolo ; mosaic
    argument:
    --yolo_path:yolo path include images and labels document
    --mosaic_path:mosaic path include labels document
    --img_path:img path include after mosaic images
    --label_path:label path include after mosaic labels
    --input_size:input size of image
    """
    # TODO:yolo2mosaic label, 据自己情况使用
    yolo2mosaic(args.yolo_path, args.mosaic_path)

    mosaic_indexes = os.listdir(args.mosaic_path)

    for j, item in enumerate(tqdm(mosaic_indexes), start=1):
        with open(os.path.join(args.mosaic_path, item), 'r') as f:
            annotation = f.readlines()
            # TODO:yolo2mosaic method
            out_img, out_label = mosaic(annotation)
            # TODO:save mosaic2yolo label
            mosaic2yolo(out_img, out_label, j, args.img_path, args.label_path)

    print(f'The mosaic format label has been converted to yolo format label and img and saved!')