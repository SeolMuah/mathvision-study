# MathVision - Python version

python >= 3.8

## 강체의 좌표 변환
main.py
```
기준 강체의 세점과 회전 강체의 세점이 주어 졌을때 변환행렬 및 변화된 지점 구하기

get_rotation_theta(r1, r2) 
    : 두 벡터가 주어졌을때 두 벡터의 회전각도
get_rotation_matrix(vec1 : np.array, vec2 : np.array) 
    : 두 벡터와  주어졌을 때 회전행렬
get_rotation_pos(rotation_pos, ref_pos, com_pos, R1, R2=None) 
    : 기준 강체의 변화 시키려는 지점과, 기준 강체의 기준점, 회전 강체의 기준점, 회전행렬이 주어졌을 때, 변화된 지점
```
