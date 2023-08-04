from colorthief import ColorThief
import matplotlib.pyplot as plt
import colorsys
import cv2
import numpy as np

# 메인 함수
def main():
    image = cv2.imread('./test1.png') # 파일 읽어들이기

    # BGR로 색추출
    bgrLower = np.array([102, 255, 255])    # 추출할 색의 하한
    bgrUpper = np.array([102, 255, 255])    # 추출할 색의 상한
    bgrResult = bgrExtraction(image, bgrLower, bgrUpper)
    cv2.imshow('BGR_test1', bgrResult)
    sleep(1)

    # HSV로 색추출
    hsvLower = np.array([30, 153, 255])    # 추출할 색의 하한
    hsvUpper = np.array([30, 153, 255])    # 추출할 색의 상한
    hsvResult = hsvExtraction(image, hsvLower, hsvUpper)
    cv2.imshow('HSV_test1', hsvResult)
    sleep(1)

    while True:
        # 키 입력을 1ms기다리고, key가「q」이면 break
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# BGR로 특정 색을 추출하는 함수
def bgrExtraction(image, bgrLower, bgrUpper):
    img_mask = cv2.inRange(image, bgrLower, bgrUpper) 
    result = cv2.bitwise_and(image, image, mask=img_mask) 
    return result

# HSV로 특정 색을 추출하는 함수
def hsvExtraction(image, hsvLower, hsvUpper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper) 
    result = cv2.bitwise_and(image, image, mask=hsv_mask)
    return result

main()