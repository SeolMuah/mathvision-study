import sys
import numpy as np
import cv2
from enum import Enum

NORMAL = "normal"
REFLEX = "reflex"
CONCAVE = "concave"
TWIST = "twist"

def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이터

    global points, left_m, right_m, draw_on, bg, H, W

    
    if event == cv2.EVENT_LBUTTONDOWN and draw_on: # 왼쪽이 눌러지면 실행
        points.append((x,y))
        left_m.append((x,y,1))
        right_m.append((-x**2-y**2))
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP and draw_on:
        #점 찍기
        cv2.line(bg, points[-1], points[-1], (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)), 5, cv2.LINE_AA)  
        cv2.putText(bg, f"{chr(64+len(points))}'({x},{y})", (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow('polygon', bg)
        
        print(f'입력 받은 좌표: {points}') # 좌표 출력

    elif event == cv2.EVENT_RBUTTONDOWN :
        #원 완성
        if len(left_m) >= 3 :
            l_m = np.array(left_m)
            r_m = np.array(right_m)
            A_plus = np.linalg.inv(l_m.T @ l_m) @ l_m.T
            result = A_plus @ r_m
            #중심 좌표 및 반지름
            a,b,c = result
            center_x = int(-1/2 * a)
            center_y = int(-1/2 * b)
            radius = int(np.sqrt(-c + 1/4*a**2 + 1/4*b**2))
            cv2.circle(bg, (center_x, center_y), radius, (0,255,255), 1)
            cv2.putText(bg, f"the equation of circle  : (x-{center_x})^2 + (y-{center_y})^2 = {radius*2}", (5,30), 
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('polygon', bg)

            draw_on = False

    elif event == cv2.EVENT_RBUTTONDBLCLK :
        #다시 그리기
        draw_on = True
        points = []
        left_m = []
        right_m = []
        bg = np.ones((H, W, 3), dtype=np.uint8) * 0
        cv2.imshow('polygon', bg)
    

draw_on = True
points = []
left_m = []
right_m = []

# 흰색 배경 이미지 생성
H, W = 480, 640
bg = np.ones((H, W, 3), dtype=np.uint8) * 0

# 윈도우 창
cv2.namedWindow('polygon')

# 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
# 마우스 이벤트가 발생하면 on_mouse 함수 실행
cv2.setMouseCallback('polygon', on_mouse, bg)

# 영상 출력
cv2.imshow('polygon', bg)
cv2.waitKey()

cv2.destroyAllWindows()