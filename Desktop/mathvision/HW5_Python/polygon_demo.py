import sys
import numpy as np
import cv2


def get_poly_area(points) :
    area = 0
    for i in range(2, len(points)) :
        area += (points[i-1][0] - points[0][0]) * (points[i][1] - points[0][1]) - \
            (points[i-1][1] - points[0][1]) * (points[i][0] - points[0][0])
        
    return abs(area) / 2


def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이터

    global points, vectors, draw_on, bg

    
    if event == cv2.EVENT_LBUTTONDOWN and draw_on: # 왼쪽이 눌러지면 실행
        points.append((x,y))
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP and draw_on:
        #선 그리기
        cv2.line(bg, points[-1], points[-1], (0, 255, 0), 3, cv2.LINE_AA)
        if len(points) > 1 : 
            cv2.line(bg, points[-2], points[-1], (255, 0, 0), 1, cv2.LINE_AA)
            vectors.append((np.array(points[-1]) - np.array(points[-2])))
            print(f'입력 받은 벡터: {vectors}') # 좌표 출력
     
        cv2.putText(bg, f'{chr(64+len(points))}({x},{y})', (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow('polygon', bg)
        print(f'입력 받은 좌표: {points}') # 좌표 출력

    elif event == cv2.EVENT_RBUTTONDOWN :
        #다각형 닫기 및 면적 출력
        if len(points) > 2 : 
            vectors.append((np.array(points[0]) - np.array(points[-1])))
            print(f'입력 받은 벡터: {vectors}') # 좌표 출력
            print(f'다각형의 넓이 : {get_poly_area(points)}')
            text = f'Area of polygon : {get_poly_area(points)}'
            cv2.putText(bg, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.line(bg, points[-1], points[0], (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('polygon', bg)
            draw_on = False
            
            
            
            
    elif event == cv2.EVENT_RBUTTONDBLCLK :
        #다시 그리기
        draw_on = True
        points = []
        vectors = []
        bg = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imshow('polygon', bg)
    

draw_on = True
points = []
vectors = []

# 흰색 배경 이미지 생성
bg = np.ones((480, 640, 3), dtype=np.uint8) * 255

# 윈도우 창
cv2.namedWindow('polygon')

# 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
# 마우스 이벤트가 발생하면 on_mouse 함수 실행
cv2.setMouseCallback('polygon', on_mouse, bg)


# 영상 출력
cv2.imshow('polygon', bg)
cv2.waitKey()

cv2.destroyAllWindows()