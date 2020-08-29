import numpy as np
import os
from pretreatment import mkdir

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


def change_txt_list_annotation(image_id, new_target, saveroot, id):
    dir = os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.txt')
    # mkdir(dir)
    with open(dir, 'a') as f:
        for bndbox in new_target:
            bndboxs = str(bndbox[0]) + ' ' + str(bndbox[1]) + ' ' + str(bndbox[2]) + ' ' + str(bndbox[3]) + '\n'
            f.write(bndboxs)



if __name__ == '__main__':

    image_id = 'test0'
    new_target = [[1, 2, 3, 4], [2, 2, 2, 2], [3, 3, 3, 3]]
    saveroot = "write_test"
    mkdir(saveroot)
    id = 0
    change_txt_list_annotation(image_id, new_target, saveroot, id)
