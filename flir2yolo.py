import argparse
import os
import json
import glob

# TODO:
"""
path
    data
        img.jpg
    coco.json

out ann path 
    img.txt

out img path 
    rename img.jpg
"""
if __name__ == '__main__':
    # linux命令行打印参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', default=None,
        help="dir containing img and json"
    )
    parser.add_argument(
        '--out_ann_path', help="txt dir",
        default=None
    )
    parser.add_argument(
        '--out_img_path', help="img new dir",
        default=None
    )
    args = parser.parse_args()

    # glob获取所有匹配的json路径
    json_path = sorted(glob.glob(os.path.join(args.path, '*coco.json')))

    for json_file in json_path:
        with open(json_file) as f:
            # 将json—file转为字典
            data = json.load(f)
            images = data['images']
            annotations = data['annotations']

            # 图片的w，h为了后续归一化
            width = 640
            height = 512

            for img in images:
                converted_result = []
                for ann in annotations:
                    # 只训练三类：人，车，自行车，后面想增加也可以改
                    if ann['image_id'] == img['id'] and ann['category_id'] <= 3:
                        class_id = int(ann['category_id'])

                        # 将bbox转化为float，flir是coco格式是左上坐标
                        left, top, bbox_w, bbox_h = map(float, ann['bbox'])

                        # yolo 的id是从0开始，flir从1开始
                        class_id -= 1

                        # 求中心坐标
                        x_center, y_center = (
                            left + bbox_w / 2, top + bbox_h / 2
                        )

                        # 归一化中心坐标
                        x, y = (x_center / width, y_center / height)
                        w, h = (bbox_w / width, bbox_h / height)

                        converted_result.append(
                            (class_id, x, y, w, h)
                        )

                image_id = str(img['id']).zfill(5)
                image_name = "FLIR_" + str(image_id)
                image_save = image_name + ".jpg"

                print(image_save)

                # 重命名图片data，这里我用的是备份文件夹data
                os.rename(os.path.join(args.path, img['file_name']),
                          os.path.join(args.out_img_path, image_save))

                # 写入yolo的txt文件
                file = open(args.out_ann_path + image_name + ".txt", 'w+')
                file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % data for data in converted_result))
                file.close()
