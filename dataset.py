import numpy as np
import os
import cv2
import pandas as pd

class DatasetTensor:    
    def __init__(self,dir_name,sigma,stepY,stepX):
        self.dir_name = dir_name
        self.files = self.find_files(dir_name)
        self.sigma = sigma
        self.stepY = stepY
        self.stepX = stepX
    
    def find_files(self, dir_name):
        massiv = []
        for file in os.listdir(dir_name):
            if file.endswith('.jpg') or file.endswith('.png'):
                massiv.append(file)           
        return massiv
    
    def set_train_mode(self):
        self.test_mode = False
        self.length = self.length_train

    def set_test_mode(self):
        self.test_mode = True
        self.length = self.length_test
    
    def __len__(self):
        result = len(self.files)
        return int(result)

    def __getitem__(self, idx):
        file =self.files[idx]
        image  = cv2.imread(self.dir_name+file)  
        df = pd.read_csv(self.dir_name+file[:-3]+'txt', sep = ' ', header = None)
       # print(df)
        df.columns = ['class_id', 'x', 'y', 'width', 'height']
        tensor = []
        e = 2.718
        Sigma=self.sigma
        for j in np.arange(0,1,self.stepY):
            spisok2 = []
            for i in np.arange(0,1,self.stepX):
                MinX = None
                MinY = None
                minR = 1
                for c in range(len(df)):
                    attr1 =  df.iloc[c][['x', 'y', 'width', 'height', 'class_id']]
                    x = attr1[0]
                    y = attr1[1]
                    x1 = x-i
                    y1 = y-j
                    pifR = ((x1**2) + (y1**2))**(1/2)
                    if pifR < minR:
                            minR = pifR
                            MinX = x1
                            MinY = y1
                            Minq1 = attr1[2]
                            Minq2 = attr1[3]
                spisok2.append([MinX,MinY,Minq1,Minq2,
                                            np.exp(-(((MinX**2)+(MinY**2)) )/Sigma**2)])
            tensor.append(spisok2) 
 
        return np.array(tensor),np.array(image)/255.0
