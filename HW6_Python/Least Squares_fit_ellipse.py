import sys
import numpy as np
import cv2
from enum import Enum

NORMAL = "normal"
REFLEX = "reflex"
CONCAVE = "concave"
TWIST = "twist"

def num_to_str(num) :
    sign =  '+' if num>0 else '-'
    return sign + f'{abs(num):3.1f}'
def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이터

    global points, left_m, right_m, draw_on, bg, H, W

    
    if event == cv2.EVENT_LBUTTONDOWN and draw_on: # 왼쪽이 눌러지면 실행
        points.append((x,y))
        left_m.append((x**2,x*y,y**2,x,y,1))
    
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP and draw_on:
        #점 찍기
        cv2.line(bg, points[-1], points[-1], (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)), 5, cv2.LINE_AA)  
        cv2.putText(bg, f"{chr(64+len(points))}'({x},{y})", (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow('polygon', bg)
        
        print(f'입력 받은 좌표: {points}') # 좌표 출력

    elif event == cv2.EVENT_RBUTTONDOWN :
        #타원 완성
        if len(left_m) >= 3 :
            l_m = np.array(left_m)
            U,s,Vt = np.linalg.svd(l_m, full_matrices=True)
            print(f"특이값 : {s}")
            print(f"우측 특이벡터 : {Vt}")
            result = Vt[-1] #가장 마지막 값이 특이값이 가장 작은 것
            a,b,c,d,e,f = result
            print(f"A*x = {l_m@result}")

            #중심 좌표 
            center_x = (2*c*d-b*e) / (b**2-4*a*c)
            center_y = (2*a*e-b*d) / (b**2-4*a*c)
   
            #회전 각
            theta = np.arctan2(b, a-c)/2 

            #고유값
            e1 = a*np.cos(theta)**2 + b*np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2
            e2 = a*np.sin(theta)**2 - b*np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2

            #스케일 역수값
            scale_inv = (a*center_x**2 + b*center_x*center_y + c*center_y**2) - f

            #장축 및 단축 길이
            l_length = scale_inv/e1
            s_length = scale_inv/e2
            if l_length > 0 and s_length > 0 : 
                l_length = np.sqrt(l_length)
                s_length = np.sqrt(s_length)
                center_x, center_y, theta, l_length, s_length = int(center_x), int(center_y), int(theta* 180 / np.pi), int(l_length), int(s_length)
                cv2.ellipse(bg, (center_x, center_y), (l_length, s_length), theta, 0, 360, (0,255,255), 1)
                cv2.imshow('polygon', bg)
            else :
                print("오류~!")

            

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