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
source = f'{root_dir}/test1.jpg'

img = cv2.imread(source)
# 모델 예측하기
# result_top은 class가 상의인 것들만 예측
# result_pants는 class가 하의인 것들만 예측
results_top = model.predict(img, imgsz = 320, show= True, classes = [0,1,2,4,5,6,8,10,11,12,17,18,20,21], 
              boxes=True)
results_pants = model.predict(img, imgsz = 320, show= True, classes = [3,9,13,14,15,16], 
              boxes=True)
# names에 모델의 class를 저장한다.
names = model.names
# results_top에서 검출된 class를 출력
for r in results_top:
    for c in r.boxes.cls:
        print(names[int(c)])
        
# for box in boxes :
#     print(box.xyxy.cpu().detach().numpy().tolist())
#     print(box.conf.cpu().detach().numpy().tolist())
#     print(box.cls.cpu().detach().numpy().tolist())

for result in results_top:
    boxes_top = result.boxes.cpu().numpy()
    for i, box in enumerate(boxes_top):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        cv2.resize(crop, dsize=(250,300))
        if results_top:
            for c in result.boxes.cls:
                cv2.imwrite("save_top/" + names[int(c)] + str(i) + ".jpg", crop)

for result in results_pants:
    boxes_pants = result.boxes.cpu().numpy()
    for i, box in enumerate(boxes_pants):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        if results_top:
            for c in result.boxes.cls:
                cv2.resize(crop, dsize=(250,320))
                cv2.imwrite("save_pants/" + names[int(c)]+ str(i) + ".jpg", crop)

plots_top = results_top[0].plot()
cv2.imshow("plot", plots_top)
plots_pants = results_pants[0].plot()
cv2.imshow("plot", plots_pants)
if cv2.waitKey(1)&0xFF == 27: # esc 누르면 닫음
        break
cv2.destroyAllWindows()