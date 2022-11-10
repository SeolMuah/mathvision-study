import sys
import numpy as np
import cv2
from enum import Enum

RED = (0,0,255)
ORANGE = (0,50,255)
YELLOW = (0,255,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
NAVY = (255,51,0)
PURPLE = (255,0,100)
WHITE = (255,255,255)
RAINBOW = [RED, ORANGE, YELLOW, GREEN, BLUE, NAVY, PURPLE]
WINDOW_H, WINDOW_W = 480*2, 640*2
pre_bg = 0
estimation_on = False

def robust_parameter_estimation(A,y,iter=10) :
    A_plus = np.linalg.inv(A.T @ A) @ A.T
    p = A_plus @ y

    params =[]
    points = []
    for _ in range(iter) :
        r = y - A @ p
        W = np.diag(1 / (np.abs(r)/1.3998 + 1))
        p = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
        a, b = p
        x1, x2 = 0, WINDOW_W
        y1, y2 = int(a*x1+b), int(a*x2+b)
        params.append(p)
        points.append(((x1,y1), (x2, y2)))

    return params, points


def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이터

    global points, v2_left_m, v2_right_m, v3_left_m, draw_on, bg, H, W

    
    if event == cv2.EVENT_LBUTTONDOWN and draw_on: # 왼쪽이 눌러지면 실행
        points.append((x,y))
        v2_left_m.append((x,1))
        v2_right_m.append((y))
        v3_left_m.append((x,y,1))
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP and draw_on:
        #점 찍기
        cv2.line(bg, points[-1], points[-1], (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)), 5, cv2.LINE_AA)  
        cv2.putText(bg, f"{chr(64+len(points))}'({x},{y})", (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, WHITE, 1, cv2.LINE_AA)
        cv2.imshow(name, bg)
        print(f'입력 받은 좌표: {points}') # 좌표 출력
    # elif event == cv2.EVENT_LBUTTONDOWN and estimation_on:
        

    elif event == cv2.EVENT_RBUTTONDOWN :
        #근사 직선 그리기
        if len(points) >= 2 :
            l_m = np.array(v2_left_m)
            r_m = np.array(v2_right_m)

            #(1) y=ax+b의 해구하기
            A_plus = np.linalg.inv(l_m.T @ l_m) @ l_m.T
            result = A_plus @ r_m
            a,b = result
            print(f'y={a}x+{b}')
            cv2.putText(bg, f'LS : y={a:.4f}x+{b:.4f}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, WHITE, 1, cv2.LINE_AA)

            #두점 구하기
            x1, x2 = 0, WINDOW_W
            y1, y2 = int(a*x1+b), int(a*x2+b)
            cv2.line(bg, (x1,y1), (x2,y2), WHITE, 1, cv2.LINE_AA)

            #Robust Param Estimation
            estimation_on = True
            params, points = robust_parameter_estimation(l_m, r_m, 7)
            for i, ps in enumerate(points): 
                a, b = params[i]
                cv2.putText(bg, f'Robust LR{i} : y={a:.4f}x+{b:.4f}', (10, 20 + (i+1)*30), cv2.FONT_HERSHEY_PLAIN, 1.0, RAINBOW[i%7], 1, cv2.LINE_AA)
                cv2.line(bg, ps[0], ps[1], RAINBOW[i%7], 1, cv2.LINE_AA)

            cv2.imshow(name, bg)
            draw_on = False

    elif event == cv2.EVENT_RBUTTONDBLCLK :
        #다시 그리기
        draw_on = True
        points = []
        v2_left_m = []
        v2_right_m = []
        v3_left_m = []
        bg = np.ones((WINDOW_H, WINDOW_W, 3), dtype=np.uint8) * 0
        cv2.imshow(name, bg)
    

draw_on = True
points = []
v2_left_m = []
v2_right_m = []
v3_left_m = []

# 흰색 배경 이미지 생성
bg = np.ones((WINDOW_H, WINDOW_W, 3), dtype=np.uint8) * 0

# 윈도우 창
name = 'Fit Line'
cv2.namedWindow(name)

# 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
# 마우스 이벤트가 발생하면 on_mouse 함수 실행
cv2.setMouseCallback(name, on_mouse, bg)

# 영상 출력
cv2.imshow(name, bg)
cv2.waitKey()

cv2.destroyAllWindows()