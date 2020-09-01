import numpy as np
import os
from pretreatment import mkdir
import json


if __name__ == '__main__':

    JSON_DIR = "Json2XML/Json"
    # mkdir(JSON_DIR)
    for root, sub_folders, files in os.walk(JSON_DIR):

        for name in files:
            path = os.path.join(root, str(name))
            with open(path, 'r') as load_f:
                json_load = json.load(load_f)
            # img_name = json_load['images']
            img_annotation = json_load['annotations']
