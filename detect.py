import glob
import yaml
from IPython.display import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO

root_dir = "/Users/woongjae/Desktop/gradu/project"
save_pants = f'{root_dir}/runs/detect/save_pants/crops'
save_top = f'{root_dir}/runs/detect/save_top/crops'

model = YOLO('/Users/woongjae/Desktop/gradu/project/runs/detect/train4/weights/best.pt')
#img = f'{root_dir}/test1.jpg'

img = cv2.imread('/Users/woongjae/Downloads/te.jpeg')
# 모델 예측하기
results= model.predict(img, show= True, classes = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,20,21], 
              boxes=True,line_width=2, show_labels=True, conf=0.35)
# result_top은 class가 상의인 것들만 예측
# results_top = model.predict(img, show= True, classes = [0,1,2,4,5,6,8,10,11,12,17,18,20,21], 
#               boxes=True,line_width=2, show_labels=True, conf=0.35)
# result_pants는 class가 하의인 것들만 예측
# results_pants = model.predict(img, show= True, classes = [3,9,13,14,15,16], 
#               boxes=True,line_width=2, show_labels=True,conf=0.5)
# names에 모델의 class를 저장한다.
names = model.names
brand_list=[]
# results에서 검출된 class를 출력
for r in results:
    for c in r.boxes.cls:
        print(names[int(c)])
        brand_list.append(names[int(c)])
print(brand_list)

for result in results:
    boxes = result.boxes.cpu().numpy()
    for i, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        if(brand_list[i]=="pants" or brand_list[i]=="jean" or brand_list[i]=="short" or 
           brand_list[i]=="shorts" or brand_list[i]=="skirt" or brand_list[i]=="slacks"):
            cv2.resize(crop, dsize=(250,300))
            cv2.imwrite("save_pants/" + brand_list[i]+ str(i) + ".jpg", crop)
        else:
            cv2.resize(crop, dsize=(250,3200))
            cv2.imwrite("save_top/" + brand_list[i]+ str(i) + ".jpg", crop)
        

# 상의인 것들은 상의 폴더에 저장하기
# for result in results_top:
#     boxes_top = result.boxes.cpu().numpy()
#     for i, box in enumerate(boxes_top):
#         r = box.xyxy[0].astype(int)
#         crop = img[r[1]:r[3], r[0]:r[2]]
#         cv2.resize(crop, dsize=(250,300))
#         cv2.imwrite("save_top/" + brand_list[i]+ str(i) + ".jpg", crop)

# 하의는 하의 폴더에 저장하기
# for result in results_pants:
#     boxes_pants = result.boxes.cpu().numpy()
#     for i, box in enumerate(boxes_pants):
#         r = box.xyxy[0].astype(int)
#         crop = img[r[1]:r[3], r[0]:r[2]]
#         cv2.resize(crop, dsize=(250,320))
#         cv2.imwrite("save_pants/" + names[int(c)]+ str(i) + ".jpg", crop)

# plots_top = results_top[0].plot()
# cv2.imshow("plot", plots_top)
# plots_pants = results_pants[0].plot()
# cv2.imshow("plot", plots_pants)

#if cv2.waitKey(1)&0xFF == 27: # esc 누르면 닫음
cv2.waitKey(0)
cv2.destroyAllWindows()

