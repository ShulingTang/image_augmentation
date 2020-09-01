import xml.etree.ElementTree as ET
import pickle

import os
from os import getcwd
import numpy as np
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa
from pretreatment import *

ia.seed(1)



if __name__ == "__main__":

    # IMG_DIR = "公开数据集-ship_detection_online/JPEGImages"
    # XML_DIR = "公开数据集-ship_detection_online/Annotations_new"
    IMG_DIR = "test/img"
    XML_DIR = "test/annotation"

    AUG_TXT_DIR = "test/Annotations_txt_Aug"  # 存储增强后的txt文件夹路径
    mkdir(AUG_TXT_DIR)

    # AUG_XML_DIR = "公开数据集-ship_detection_online/Annotations_Aug"  # 存储增强后的XML文件夹路径
    AUG_XML_DIR = "test/Annotations_Aug"  # 存储增强后的XML文件夹路径
    mkdir(AUG_XML_DIR)

    # AUG_IMG_DIR = "公开数据集-ship_detection_online/JPEGImages_Aug"  # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = "test/JPEGImages_Aug"  # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)

    RESIZE_IMG_DIR = "test/img_resize" # 存储调整图片尺寸后的图片
    mkdir(RESIZE_IMG_DIR)

    AUGLOOP = 10  # 每张影像增强的数量
    img_size = [800, 800]  # 图片目标调整大小
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
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    for root, sub_folders, files in os.walk(XML_DIR):

        for name in files:

            bndbox = read_xml_annotation(XML_DIR, name)  # 从XML文件中读取bndbox
            # 如果不需要调整图片大小，请注释掉 192 - 195行, 将181替换180
            # 调整对应图片大小
            resize_img(IMG_DIR, name[:-4] + '.jpg', img_size, RESIZE_IMG_DIR)
            # 调整bndbox值
            bndbox = np.array(bndbox)
            bndbox = bndbox * (800 / 256)
            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机


                # 读取图片
                # img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))  # 不需改变图片大小
                img = Image.open(os.path.join(RESIZE_IMG_DIR, name[:-4] + '.jpg'))  # 需要改变图片大小
                img = np.array(img)

                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
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
                # change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, epoch)
                print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                new_bndbox_list = []

