import xml.etree.ElementTree as ET
import pickle
import cv2
import os
from os import getcwd
import numpy as np
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa
from pretreatment import *

ia.seed(1)




if __name__ == "__main__":

    IMG_DIR = "test/img"  #输入图片地址
    TXT_DIR = "test/annotation_txt"  # txt标签存储地址

    AUG_TXT_DIR = "test/Annotations_txt_Aug"  # 存储增强后的txt文件夹路径
    mkdir(AUG_TXT_DIR)

    AUG_IMG_DIR = "test/JPEGImages_Aug"  # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 10  # 每张影像增强的数量
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

    for root, sub_folders, files in os.walk(TXT_DIR):

        for name in files:
            bndbox = read_txt_annotation(TXT_DIR, name)  # 从TXT文件中读取bndbox
            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))  # 不需改变图片大小
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
                # change_xml_list_annotation(TXT_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, epoch)
                print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                new_bndbox_list = []

