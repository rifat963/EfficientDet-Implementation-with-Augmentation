import numpy as np
import pandas as pd
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def create_coco_annotations_with_bbox(train_label_bbox,
                            outputfile,
                            config_parm):
    """
    Structure of train_label_bbox contains following structure
       image_id  width  height                         bbox
    0  b6ab77fd7   1024    1024   [834.0, 222.0, 56.0, 36.0]
    1  b6ab77fd7   1024    1024  [226.0, 548.0, 130.0, 58.0]
    2  b6ab77fd7   1024    1024  [377.0, 504.0, 74.0, 160.0]
    """
    df = pd.read_csv(train_label_bbox)
    print('-- load train labels bounding boxes --')
    df['bbox'] = df['bbox'].apply(lambda x: x[1:-1].split(","))
    df['x'] = df['bbox'].apply(lambda x: x[0]).astype('float32')
    df['y'] = df['bbox'].apply(lambda x: x[1]).astype('float32')
    df['w'] = df['bbox'].apply(lambda x: x[2]).astype('float32')
    df['h'] = df['bbox'].apply(lambda x: x[3]).astype('float32')
    df = df[['image_id', 'x', 'y', 'w', 'h']]
    image_ids = df['image_id'].unique()
    image_dict = dict(zip(image_ids, range(len(image_ids))))
    print(len(image_dict))

    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    for image_id in image_ids:
        image = {'file_name': image_id + '.jpg',
                 'height': config_parm['height'],
                 'width': config_parm['width'],
                 'id': image_dict[image_id]}
        json_dict['images'].append(image)

    categories = {'supercategory': config_parm['categories']['supercategory'],
                  'id': config_parm['categories']['id'],
                  'name': config_parm['categories']['name']}
    json_dict['categories'].append(categories)

    for idx, box_id in df.iterrows():
        image_id = image_dict[box_id['image_id']]

        ann = {'area': box_id['w'] * box_id['h'],
               'iscrowd': 0,
               'image_id': image_id,
               'bbox': [box_id['x'], box_id['y'], box_id['w'], box_id['h']],
               'category_id': config_parm['categories']['id'],
               'id': idx,
               'segmentation': []}

        json_dict['annotations'].append(ann)

    print('-- write coco annotation to file: {} --'.format(outputfile))
    annFile = outputfile
    json_fp = open(annFile, 'w', encoding='utf-8')
    json_str = json.dumps(json_dict, cls=NpEncoder)
    json_fp.write(json_str)
    json_fp.close()


if __name__== '__main__':
    train_labels = '../custom-dataset/train.csv'
    output_file = '../custom-dataset/coco_annotation/instances_Images.json'
    config_parm = {
        'height': 1024,
        'width': 1024,
        'categories':{
            'supercategory': 'wh',
            'id' : 1,
            'name': 'wh'}
    }
    create_coco_annotations_with_bbox(train_labels,output_file,config_parm)









