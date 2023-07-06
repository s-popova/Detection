#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
import cv2
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_size, classes_file):
        self.data_path = data_path
        self.image_size = image_size
        self.classes = []
        with open(classes_file, "r") as f:
            for line in f.readlines():
                self.classes.append(line.strip())

        self.image_files = []
        self.annotations = []
        with open(os.path.join(data_path, "train.txt"), "r") as f:
            for line in f.readlines():
                self.image_files.append(line.strip())
                annotation_file = os.path.join(data_path, "labels", os.path.splitext(line.strip())[0] + ".txt")
                with open(annotation_file, "r") as af:
                    annotation = []
                    for annotation_line in af.readlines():
                        class_id, x_center, y_center, width, height = map(float, annotation_line.strip().split())
                        annotation.append([class_id, x_center, y_center, width, height])
                    self.annotations.append(annotation)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_path, "images", image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        annotation = self.annotations[idx]
        num_objects = len(annotation)

        boxes = torch.zeros((num_objects, 4))
        labels = torch.zeros((num_objects,))
        for i, annot in enumerate(annotation):
            class_id, x_center, y_center, width, height = annot
            x_min = (x_center - width / 2) * self.image_size
            y_min = (y_center - height / 2) * self.image_size
            x_max = (x_center + width / 2) * self.image_size
            y_max = (y_center + height / 2) * self.image_size
            boxes[i] = torch.tensor([x_min, y_min, x_max, y_max])
            labels[i] = torch.tensor(class_id)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        return image, target

