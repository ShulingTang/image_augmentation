import os
from pretreatment import mkdir
import json


if __name__ == '__main__':

    JSON_DIR = "Json2XML/Json"
    Json_name = 'train_test2017.json'

    path = os.path.join(JSON_DIR, Json_name)
    with open(path, 'r') as load_f:
        json_load = json.load(load_f)
        # img_name = json_load['images']
        img_annotation = json_load['annotations']
    count, widths, highs = 0, 0, 0
    for item in img_annotation:
        count += 1
        bbox = item['bbox']
        high = bbox[3]
        width = bbox[2]
        widths += width
        highs += high
    print('high=%s\n width=%s' % (highs / count, widths / count))
