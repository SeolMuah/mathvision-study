import numpy as np
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter()

x = np.linspace(-1, 5, 100)
y = np.linspace(-3, 4, 100)
x1, y1 = np.meshgrid(x,y)

z1 = np.sin(x1+y1-1) + (x1-y1-1)**2 - 1.5*x1 + 2.5*y1 + 1

#1) 곡면 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, y1, z1, rstride=2,cstride=2,
            cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True) 
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.title("Polynomial Surface")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


#2) Gradient Decent 법
from sympy import symbols, diff, Matrix, solve
import sympy
x,y = symbols('x,y')
f = sympy.sin(x+y-1) + (x-y-1)**2 -1.5*x + 2.5*y + 1
dfdx = diff(f,x)
dfdy = diff(f,y)
Graient = Matrix([dfdx, dfdy])
print(Graient)
lam = 0.01
x_cur = 0
y_cur = 4
x_points = []
y_points = []
z_points = []
for i in range(1000) :
    x_points.append(x_cur)
    y_points.append(y_cur)
    z_points.append(np.sin(x_cur+y_cur-1) + (x_cur-y_cur-1)**2 - 1.5*x_cur + 2.5*y_cur + 1)
    print(f"{i}: ", x_points[-1], y_points[-1], z_points[-1])
    x_next = x_cur - lam * (2*x_cur-2*y_cur+np.cos(x_cur+y_cur-1)-3.5)
    y_next = y_cur - lam * (-2*x_cur+2*y_cur+np.cos(x_cur+y_cur-1)+4.5)

    if np.abs(x_next-x_cur) < 1e-4 and np.abs(y_next-y_cur) < 1e-4 :
        x_cur = x_next
        y_cur = y_next
        break
    x_cur = x_next
    y_cur = y_next
print(f"Step size : {i}")
x_points.append(x_next)
y_points.append(y_next)
z_points.append(np.sin(x_next+y_next-1) + (x_next-y_next-1)**2 - 1.5*x_next + 2.5*y_next + 1)


#2-1) Gradient Decent 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, y1, z1, rstride=2,cstride=2,
            cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True) 
ax.scatter(x_points, y_points, z_points, s=20, c='r')
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.title("Polynomial Surface")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


################################################
#3) Newton 법
from sympy import symbols, diff, Matrix, solve
import sympy
x,y = symbols('x,y')
f = sympy.sin(x+y-1) + (x-y-1)**2 -1.5*x + 2.5*y + 1
dfdx = diff(f,x)
dfdy = diff(f,y)
dfdxdx = diff(dfdx,x)
dfdydy = diff(dfdy,y)
print(dfdxdx)
print(dfdydy)

x_cur = 0
y_cur = 4
x_points = []
y_points = []
z_points = []
for i in range(1000) :
    x_points.append(x_cur)
    y_points.append(y_cur)
    z_points.append(np.sin(x_cur+y_cur-1) + (x_cur-y_cur-1)**2 - 1.5*x_cur + 2.5*y_cur + 1)
    print(f"{i}: ", x_points[-1], y_points[-1], z_points[-1])
    x_next = x_cur - (2*x_cur-2*y_cur+np.cos(x_cur+y_cur-1)-3.5) / (2-np.sin(x_cur+y_cur-1))
    y_next = y_cur - (-2*x_cur+2*y_cur+np.cos(x_cur+y_cur-1)+4.5) / (2-np.sin(x_cur+y_cur-1))

    if np.abs(x_next-x_cur) < 1e-4 and np.abs(y_next-y_cur) < 1e-4 :
        x_cur = x_next
        y_cur = y_next
        break
    x_cur = x_next
    y_cur = y_next
print(f"Step size : {i}")
x_points.append(x_next)
y_points.append(y_next)
z_points.append(np.sin(x_next+y_next-1) + (x_next-y_next-1)**2 - 1.5*x_next + 2.5*y_next + 1)


#3-1) Gradient Decent 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, y1, z1, rstride=2,cstride=2,
            cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True) 
ax.scatter(x_points, y_points, z_points, s=20, c='r')
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.title("Polynomial Surface")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()