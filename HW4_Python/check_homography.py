import sys
import numpy as np
import cv2
from enum import Enum

NORMAL = "normal"
REFLEX = "reflex"
CONCAVE = "concave"
TWIST = "twist"

def check_homography(ref_points, compared_points) :
    
    poly_num = len(ref_points)
    ref_points = np.array(ref_points)
    compared_points = np.array(compared_points)
    
    ref_vecs = np.roll(ref_points, shift=-1, axis=0) - ref_points
    com_vecs = np.roll(compared_points, shift=-1, axis=0) - compared_points
    
    ref_cross = np.cross(ref_vecs, np.roll(ref_vecs, shift=-1, axis=0))
    com_cross = np.cross(com_vecs, np.roll(com_vecs, shift=-1, axis=0))
    
    result = (ref_cross * com_cross < 0).sum()
    
    #normal : 도형의 외적 부호가 reference 도형의 외적 부호와 모두 같은 경우
    if result == 0 :
        return NORMAL
    #reflex : 도형의 외적 부호가 reference 도형의 외적 부호와 모두 다른 경우
    elif result == poly_num :
        return REFLEX
    #twist : 자기 자신의 외적 부호가 균등하게(짝수개) 다른 경우 
    elif result%2 == 0 :
        return TWIST
    #concave : 자기 자신의 외적 부호가 홀수개 만큼 다른 경우
    else :
        return CONCAVE
    
    
def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이터

    global points, vectors, draw_on, bg, ref_rect, colors, H, W

    
    if event == cv2.EVENT_LBUTTONDOWN and draw_on: # 왼쪽이 눌러지면 실행
        points.append((x,y))
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP and draw_on:
        #선 그리기
        cv2.line(bg, points[-1], points[-1], colors[len(points)-1], 5, cv2.LINE_AA) #점 그리기
        if len(points) > 1 : 
            cv2.line(bg, points[-2], points[-1], (255, 100, 10), 1, cv2.LINE_AA)
            vectors.append((np.array(points[-1]) - np.array(points[-2])))
            print(f'입력 받은 벡터: {vectors}') # 좌표 출력

        if len(points) >= 4 :
            cv2.line(bg, points[-1], points[0], (255, 100, 10), 1, cv2.LINE_AA)
            draw_on = False
            #--------------------
            for p1, p2, c in zip(ref_rect, points, colors) :
                cv2.line(bg, p1, p2, c, 1, cv2.LINE_AA)

            result = check_homography(ref_rect, points)
            cv2.putText(bg, result, (10,H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
        cv2.putText(bg, f"{chr(64+len(points))}'({x},{y})", (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow('polygon', bg)
        
        
        
        
        
        print(f'입력 받은 좌표: {points}') # 좌표 출력



    elif event == cv2.EVENT_RBUTTONDBLCLK :
        #다시 그리기
        draw_on = True
        points = []
        vectors = []
        bg = np.ones((H, W, 3), dtype=np.uint8) * 0
        cv2.rectangle(bg, ref_rect[0], ref_rect[-2], (0,0,255), 1, cv2.LINE_AA)
        for i in range(len(ref_rect)) :
            cv2.line(bg, ref_rect[i], ref_rect[i], colors[i], 5, cv2.LINE_AA) 
            cv2.putText(bg, f"{chr(65+i)}", np.array(ref_rect[i]) + (-5, 0), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow('polygon', bg)
    

draw_on = True
points = []
vectors = []
colors = [(128,128,255), (0,255,255), (156,175,80), (170,74,181)]
# 흰색 배경 이미지 생성
H, W = 480, 640
bg = np.ones((H, W, 3), dtype=np.uint8) * 0


#기준 사각형
start_point = np.array((10,10))
ref_width=int(W//10)
ref_rect = np.array([(0,0), (0, ref_width), (ref_width, ref_width), (ref_width, 0), ]) + start_point
cv2.rectangle(bg, ref_rect[0], ref_rect[-2], (0,0,255), 1, cv2.LINE_AA)
for i in range(len(ref_rect)) :
    cv2.line(bg, ref_rect[i], ref_rect[i], colors[i], 5, cv2.LINE_AA) 
    cv2.putText(bg, f"{chr(65+i)}", np.array(ref_rect[i]) + (-5, 0), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.LINE_AA)
# 윈도우 창
cv2.namedWindow('polygon')

# 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
# 마우스 이벤트가 발생하면 on_mouse 함수 실행
cv2.setMouseCallback('polygon', on_mouse, bg)


# 영상 출력
cv2.imshow('polygon', bg)
cv2.waitKey()

cv2.destroyAllWindows()