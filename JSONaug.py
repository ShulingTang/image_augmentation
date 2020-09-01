import numpy as np
import os
from pretreatment import mkdir
import json
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
from pretreatment import *


# =========================
#
#
#
if __name__ == '__main__':

    JSON_DIR = "HRSID_JPG/annotations"  # Json文件目录
    JSON_NAME = 'train_test2017.json'  # 可选：train_test2017.json; test2017.json; train2017.json
    IMG_DIR = "HRSID_JPG/JPEGImages"

    AUG_TXT_DIR = "jsontest/Annotations_Aug"  # 存储增强后的txt文件夹路径
    mkdir(AUG_TXT_DIR)

    AUG_IMG_DIR = "jsontest/JPEGImages_Aug"  # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 3  # 每张影像增强的数量
    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []


    # 影像增强
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)), # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.35, 0.5),
            rotate=(-30, 30)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])


    path = os.path.join(JSON_DIR, str(JSON_NAME))

    # 加载json文件
    with open(path, 'r') as load_f:
        json_load = json.load(load_f)
    img_names = json_load['images']  # 图列表
    img_annotation = json_load['annotations']  # 标签列表
    for item1 in img_names:
        name = item1['file_name']
        img_id = item1['id']
        bndbox_list = []
        #  遍历标签列表，将所有指向图：img_id的所有标签中的bbox存储在bndbox_list
        for item2 in img_annotation:
            if item2['image_id'] == img_id:
                bbox = item2['bbox']  # [x1, y1, width, high]
                bndbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                # bndbox = [bbox['0'], bbox['1'], bbox['2'], bbox['3']]
                # print(bndbox)
                bndbox_list.append(bndbox) #用于存储 图id 对应的所有的bndbox
            if item2['image_id'] > img_id:
                break
        a = bndbox_list
        b = os.path.join(IMG_DIR, name)
        for epoch in range(AUGLOOP):
            seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

            # 读取图片
            img = Image.open(os.path.join(IMG_DIR, name))  # 不需改变图片大小
            img = np.array(img)

            # bndbox 坐标增强
            for i in range(len(bndbox_list)):
                bbs = ia.BoundingBoxesOnImage([
                    ia.BoundingBox(x1=bndbox_list[i][0], y1=bndbox_list[i][1], x2=bndbox_list[i][2], y2=bndbox_list[i][3]),
                ], shape=img.shape)

                bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                boxes_img_aug_list.append(bbs_aug)

                # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                        int(bbs_aug.bounding_boxes[0].y1),
                                        int(bbs_aug.bounding_boxes[0].x2),
                                        int(bbs_aug.bounding_boxes[0].y2)])
            # 存储变化后的图片
            image_aug = seq_det.augment_images([img])[0]
            path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
            # image_auged = bbs.draw_on_image(image_aug, thickness=0)
            Image.fromarray(image_aug).save(path)

            # 存储变化后对TXT
            change_txt_list_annotation(name[:-4], new_bndbox_list, AUG_TXT_DIR, epoch)
            # 存储变化后的XML
            # change_xml_list_annotation(TXT_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, epoch)
            print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
            new_bndbox_list = []

# test: P0001_3600_4400_3600_4400