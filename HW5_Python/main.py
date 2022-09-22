import numpy as np

import sympy as s
from sympy import Symbol, solve

def get_rotation_theta(r1, r2) :
    """
    :r1 : 3X1 벡터
    :r2 : 3X1 벡터
    :return: 회전 각도 (라디안)
    """
    h = np.cross(r1, r2) #r1,r2가 이루는 평면의 법선벡터
    h_unit = h / np.linalg.norm(h)
    ux, uy, uz = h_unit
    x = Symbol('x')
    cos = s.cos(x)
    sin = s.sin(x)
    R = s.Matrix([(cos+(ux**2)*(1-cos), ux*uy*(1-cos)-uz*sin, ux*uz*(1-cos)+uy*sin),
                    (uy*ux*(1-cos) + uz*sin, cos+(uy**2)*(1-cos), uy*uz*(1-cos)-ux*sin),
                    (uz*ux*(1-cos)-uy*sin, uz*uy*(1-cos)+ux*sin, cos+(uz**2)*(1-cos))])
    r1 = s.Matrix(r1)
    r2 = s.Matrix(r2)
    eq = R*r1 - r2

    solution = [solve(eq[i]) for i in range(len(eq))]
    
    #실수 값 비교하기 위한 소수점 절사
    solution_round = [set([np.round(float(n), 4) for n in eq]) for eq in solution]
    
    #3개의 방정식 풀이에서 같은 값 하나 얻기
    result = list(solution_round[0] & solution_round[1] & solution_round[2])

    return result[0]

def get_rotation_matrix(theta, vec1, vec2) :
    """
    :param theta: 회전 각도 (라디안)
    :param vec1: 3X1 벡터
    :param vec2: 3X1 벡터
    :return: 3X3 회전 행렬
    """
    h = np.cross(vec1, vec2)
    h = h/np.linalg.norm(h)
    theta = float(theta)
    cos = np.cos(theta)
    sin = np.sin(theta)
    ux,uy,uz = h
    R = np.array([(cos+(ux**2)*(1-cos), ux*uy*(1-cos)-uz*sin, ux*uz*(1-cos)+uy*sin),
                    (uy*ux*(1-cos) + uz*sin, cos+(uy**2)*(1-cos), uy*uz*(1-cos)-ux*sin),
                    (uz*ux*(1-cos)-uy*sin, uz*uy*(1-cos)+ux*sin, cos+(uz**2)*(1-cos))])
    return R

def get_rotation_pos(rotation_pos, ref_pos, com_pos, R1, R2=None) :
    """
    :param rotation_pos: 기준 강체의 변환하곶 하는 지점 3X1 벡터
    :param ref_pos: 기준 강체의 기준점 3X1 벡터
    :param com_pos: 회전한 강체의 기준점 3X1 벡터
    :param R1 : 3X3 회전 행렬
    :param R2 : 3X3 회전 행렬
    :return: 회전 각도 라디안
    """
    
    ref_vec = rotation_pos - ref_pos
    rotation_vec = R2 @ R1 @ ref_vec if R2 is not None else R1 @ ref_vec

    return rotation_vec + com_pos



################################################################

ref_points = np.array([(-0.500000,	0.000000,	2.121320),
                (0.500000,	0.000000,	2.121320),
                (0.500000,	-0.707107,	2.828427)
])
                
com_points = np.array([
    (1.363005,	-0.427130,	2.339082),
    (1.748084,	0.437983,	2.017688),
    (2.636461,	0.184843,	2.400710)
])

ref_vecs = [ref_points[i]-ref_points[0] for i in range(len(ref_points))[1:]]
com_vecs = [com_points[i]-com_points[0] for i in range(len(com_points))[1:]]
print(f"ref_vecs : {ref_vecs}")
print(f"com_vecs : {com_vecs}")

#각 평면의 법선 벡터
h1 = np.cross(ref_vecs[0], ref_vecs[1])
h2 = np.cross(com_vecs[0], com_vecs[1])
print(f"ref_norm_vec : {h1}")
print(f"com_norm_vec : {h2}")


#(1) h1 -> h2로 회전하기 위한 Theta 구하기 
theta = get_rotation_theta(h1, h2)
print(f"Theta (Radian, degree) : ({theta:.4f}, {theta/np.pi*180:.4f})")

#(2) h1 -> h2 회전 행렬 
R1 = get_rotation_matrix(theta, h1, h2)

#검산 : h1@R == h2
print(f"R1 @ h1 = {R1 @ h1},\nh2 = {h2}") 

#(3) R1(p1p3) -> p1'p3'로 회전 회전하기 위한 Theta2 구하기
p13 = ref_vecs[-1]
r1_p13 = R1 @ p13
p13_ = com_vecs[-1]

#R1(p1p3) 과 p1'p3'이 이루는 평면의 법선 벡터
theta2 = get_rotation_theta(r1_p13, p13_)
print(f"Theta (Radian, degree) : ({theta2:.4f}, {theta2/np.pi*180:.4f})")

#(4) R1(p1p3) -> p1'p3' 회전 행렬 

R2 = get_rotation_matrix(theta2, r1_p13, p13_)

#검산 : R2 @ R1(p1p3) == p13
print(f"R2 @ R1(p1p3) = {R2 @ r1_p13},\nh2 = {p13_}") 
#------------------------------------------------------------
#p4 -> p4' 맞는지 확인

p4 = np.array(
    (0.500000,	0.707107,	2.828427)
)

p4_ = np.array(
    (1.498100,	0.871000,	2.883700)
)

p4_rotaion = get_rotation_pos(p4,  ref_points[0], com_points[0], R1, R2)
print(f"p4 rotaion pos = {p4_rotaion},\nanswer = {p4_}") 

#p5 -> p5' 좌표 구하기

p5 = np.array(
    (1.000000,	1.000000,	1.000000)
)

p5_rotaion = get_rotation_pos(p5,  ref_points[0], com_points[0], R1, R2)
print(f"p5 rotaion pos = {p5_rotaion}") 
