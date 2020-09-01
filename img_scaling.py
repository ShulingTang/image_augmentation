from pretreatment import *
import numpy as np

def resize_img(DATADIR, data_k, img_size):
    w = img_size[0]
    h = img_size[1]
    '''设置目标像素大小，此处设为300'''
    path = os.path.join(DATADIR, data_k)
    # 返回path路径下所有文件的名字，以及文件夹的名字，
    img_list = os.listdir(path)

    for name in img_list:
        if name.endswith('.jpg'):
            # scaling bbox
            bndbox = read_xml_annotation(XML_DIR, name[:-4] + '.xml')  # 从XML文件中读取bndbox
            # 调整bndbox值
            bndbox = np.array(bndbox)
            bndbox = bndbox * (800 / 256)
            # 存储变化后对TXT
            save_txt = path + '_txt/'
            change_txt_list_annotation(name[:-4], bndbox, save_txt, 0)

            # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
            img_array = cv2.imread((path + '/' + name), cv2.IMREAD_COLOR)
            # 调用cv2.resize函数resize图片
            new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)
            img_name = str(name)
            '''生成图片存储的目标路径'''
            save_path = path + '_new/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print(name)
            '''调用cv.2的imwrite函数保存图片'''
            save_img = save_path + img_name
            cv2.imwrite(save_img, new_array)

if __name__ == '__main__':
    # 设置图片路径
    DATADIR = "test/img"
    XML_DIR = "test/xml"


    mkdir(DATADIR)
    # data_k = 'query'
    data_k = ''

    # 需要修改的新的尺寸
    img_size = [800, 800]
    resize_img(DATADIR, data_k, img_size)