import os
import xml.etree.ElementTree as ET
import cv2

def mkdir(path):

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
         # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def change_txt_list_annotation(image_id, new_target, saveroot, id):
    dir = os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.txt')

    with open(dir, 'a') as f:
        for bndbox in new_target:
            bndboxs = str(bndbox[0]) + ' ' + str(bndbox[1]) + ' ' + str(bndbox[2]) + ' ' + str(bndbox[3]) + '\n'
            f.write(bndboxs)

def change_xml_list_annotation(root, image_id, new_target, saveroot, id):

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))

def read_txt_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    # a = in_file.read()
    bndboxlist = []
    for line in in_file.readlines():
        print(line)
        line = line.strip()
        bndbox = list(map(int, line.split()))
        bndboxlist.append(bndbox)
    return bndboxlist

def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin,ymin,xmax,ymax])
        print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist
# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))


def resize_img(path, filename, img_size, save_path):
    w, h = img_size[0], img_size[1]
    if filename.endswith('.jpg'):
        # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
        img_array = cv2.imread((path + '/' + filename), cv2.IMREAD_COLOR)
        # 调用cv2.resize函数resize图片
        new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)
        img_name = str(filename)
        # '''生成图片存储的目标路径'''
        # save_path = path + '_new/'
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        print(filename)
        '''调用cv.2的imwrite函数保存图片'''
        save_img = save_path + '/' + img_name
        cv2.imwrite(save_img, new_array)

