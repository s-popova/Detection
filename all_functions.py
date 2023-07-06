import numpy as np
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt

def border(tensor_in,border_1):
    result_wo_IOB=[]
    x=0
    y=0
    tensor = deepcopy(tensor_in)
    for i in tensor:
        for j in i:
            j[1] += x
            j[0] += y
            y += 1 / tensor.shape[0]
            if y >= 0.99:
                y = 0
                x += 1 / tensor.shape[1]
            if j[4]>border_1:
                result_wo_IOB.append(j)
    return np.array(result_wo_IOB)

def visual(Tensor, image ):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.shape
    (img_height,img_width)=image.shape[:2]

    for i in Tensor:
        x = int(i[0]*img_width)
        y = int(i[1]*img_height)
        w = int(i[2]*img_width)
        h = int(i[3]*img_height)

        cv2.rectangle(image_rgb, (x - int(np.round(w/2)), y - int(np.round(h/2))), (x + int(np.round(w/2)), y + int(np.round(h/2))), (0, 255, 0), 1)
        text = '{:.4}'.format(str(i[4]))
        cv2.putText(image_rgb, text, (x - int(np.round(w/2)), y - 10 - int(np.round(h/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
     
    
    return show_img_1(image_rgb)

def show_img_1(img,cmap=None):
    plt.figure(figsize=(25, 25))
    plt.imshow(img,cmap)
    plt.show()

    

    
    
