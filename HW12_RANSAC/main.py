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
WINDOW_H, WINDOW_W = 480, 640
pre_bg = 0
estimation_on = False


def ransac(A, iter=2000, T=2*10, n_sample=3) :
    
    n_data = len(A)
    max_cnt = 0
    best_model = [.0, .0, .0]

    for _ in range(iter) :
        k = np.floor(n_data*np.random.rand(n_sample)).astype(int)
        Ak = A[k]
        U,M,V= np.linalg.svd(Ak)
        pk = V[-1]
        residual = np.abs(A @ pk) / np.sqrt(pk[0]**2 + pk[1]**2) #점과 직선사이의 거리
        cnt = (residual < T).sum()

        if cnt > max_cnt :
            best_model = pk
            max_cnt = cnt
                
    residual = np.abs(A @ best_model) / np.sqrt(pk[0]**2 + pk[1]**2)
    in_k = residual < T
    U,M,V= np.linalg.svd(A[in_k])
    params = V[-1]
    
    return params


def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이터

    global points, v2_left_m, v2_right_m, v3_left_m, draw_on, bg, H, W

    
    if event == cv2.EVENT_LBUTTONDOWN and draw_on: # 왼쪽이 눌러지면 실행
        points.append((x,y))
        v2_left_m.append((x,y,1))
        v2_right_m.append((0))
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP and draw_on:
        #점 찍기
        cv2.line(bg, points[-1], points[-1], (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)), 5, cv2.LINE_AA)  
        # cv2.putText(bg, f"{chr(64+len(points))}'({x},{y})", (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, WHITE, 1, cv2.LINE_AA)
        cv2.imshow(name, bg)
        print(f'입력 받은 좌표: {points}') # 좌표 출력

    elif event == cv2.EVENT_RBUTTONDOWN :
        #근사 직선 그리기
        if len(points) >= 2  and draw_on:
            l_m = np.array(v2_left_m)
            r_m = np.array(v2_right_m)


            #Robust Param Estimation
            T = 100.0
            iter=2000
            params = ransac(l_m, iter=iter, T=T, n_sample=3)

            a, b, c = params
            x1, x2 = 0, WINDOW_W
            y1, y2 = int((-a*x1-c)/b), int((-a*x2-c)/b)

            residual = np.abs(l_m @ params) / np.sqrt(a**2 + b**2)
            in_k = residual < T

            points = np.array(points)
            for p in points[in_k]:
                cv2.line(bg, p, p, YELLOW, 5, cv2.LINE_AA)  

            for p in points[np.invert(in_k)]:
                cv2.line(bg, p, p, RED, 5, cv2.LINE_AA) 


            cv2.putText(bg, f'Ransac line : {a:.4f}x+{b:.4f}y+{c:.4f}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, GREEN, 1, cv2.LINE_AA)
            ratio = 100*np.invert(in_k).sum()/len(points)
            cv2.putText(bg, f'Ransac Parameter : Iter:{iter}, Threshold:{T}, outlier ratio:{ratio:.0f}%', (10, 40),cv2.FONT_HERSHEY_PLAIN, 1.0, GREEN, 1, cv2.LINE_AA)
            cv2.line(bg, (x1,y1), (x2,y2), GREEN, 1, cv2.LINE_AA)

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