"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import os.path as osp
import json
import random


# csv_Classes={0: 0, 1: 1}
#visdrone
JSON_Classes={'ignored_regions': 0, 'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'tricycle': 7, 'awning-tricycle': 8, 'bus': 9, 'motor': 10,'others': 11}
#pascal
# JSON_Classes={'person':0,'bird':1,'cat':2,'dog':3,'horse':4,'sheep':5,'aeroplane':6,'bicycle':7,'boat':8,'bus':9,'car':10,'motorbike':11,'train':12,'bottle':13,'chair':14,
#               'diningtable':15,'pottedplant':16,'sofa':17,'tvmonitor':18,'cow':19}



class CSVGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        train,
        base_dir=None,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir



        # parse the provided class file
        self.classes=JSON_Classes


        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self._annopath = osp.join('%s', 'labels', '%s.json')
        self._imgpath = osp.join('%s', 'images', '%s.jpg')
        self.ids=list()
        self.base_dir='/input0/visdrone_detection'
        train_num=int(len(os.listdir(os.path.join(self.base_dir,'images')))*0.9)
        all_list=os.listdir(os.path.join(self.base_dir,'images'))
        #将数组顺序随机打乱
        random.shuffle(all_list)

        if train:

            for each in all_list[:train_num]:
                img_name=each.split('.')[0]
                self.ids.append((self.base_dir,img_name))
        else:
            for each in all_list[train_num:]:
                img_name=each.split('.')[0]
                self.ids.append((self.base_dir,img_name))

        
        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        这一步是对父类的重写，在父类的group_images方法中，会根据这一步的返回值构建batch_size的数据
        """
        return len(self.ids)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
#         print('*'*10,self.labels)
        return self.labels[label]



    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        img_id=self.ids[image_index]
        image = Image.open(self._imgpath%img_id)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
#         print('Loading images...',image_index)
        img_id=self.ids[image_index]
        return read_image_bgr(self._imgpath%img_id)

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
#         print('Loading annotations...',image_index)
        img_id=self.ids[image_index]
        path=self._annopath%img_id
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        
        a=open(path)
        json_info=json.load(a)
        image_height=json_info['image_height']
        image_width=json_info['image_width']
        for i in range(int(json_info['num_box'])):
            box=json_info['bboxes'][i]
            annotations['labels']=np.concatenate((annotations['labels'],[self.name_to_label(box['label'])]))

            annotations['bboxes']=np.concatenate((annotations['bboxes'],[[float(box['x_min'])*image_width,
                                                                         float(box['y_min'])*image_height,
                                                                          float(box['x_max'])*image_width,
                                                                          float(box['y_max'])*image_height,
                                                                         ]]))



        return annotations
