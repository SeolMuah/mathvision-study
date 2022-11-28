import numpy as np
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter()

x = np.linspace(-1, 1.5, 100)
y = np.linspace(-1.2, 0.2, 100)

x1, y1 = np.meshgrid(x,y)

z1 = (x1+y1)*(x1*y1 + x1*y1**2)

#1) 곡면 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, y1, z1, rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True) 
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.title("Polynomial Surface")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


#2) (1,0) Gradient 
from sympy import symbols, diff, Matrix, solve
x,y = symbols('x,y')
f = (x+y)*(x*y + x*y**2)
ret = f.subs({x:1, y:0})
dfdx = diff(f,x)
dfdy = diff(f,y)
Graient = Matrix([dfdx, dfdy])
pp.pprint(Graient)
graient_ret = Graient.subs({x:1, y:0})
pp.pprint(graient_ret)


#3) (1,0) Gradient 시각화
x = np.ones((100))
y = np.linspace(-1.2, 0.2, 100)
z = y
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, y1, z1, rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True)
ax.plot(x,y,z, c='g')
ax.scatter([1],[0],[0], c='r', s=50)
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.title("Polynomial Surface")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#4) Hessian Test 
dfdxdx = diff(dfdx, x)
dfdxdy = diff(dfdx, y)
dfdydx = diff(dfdy, x)
dfdydy = diff(dfdy, y)
Hessian = Matrix([[dfdxdx, dfdxdy], [dfdydx, dfdydy]])
pp.pprint(Hessian)

#Critical Point 
solve_ret = solve(Graient)
print(solve_ret)

for ret in solve_ret :
    m = Hessian.subs(ret)
    eigen = list(m.eigenvals().keys())
    judge = ""
    if all([v>0 for v in eigen]) :
        judge = "극소점"
    elif all([v<0 for v in eigen]) :
        judge = "극대점"
    else :
        judge = "안장점"
    print(f"{ret}, Hessian EigenValue:{eigen}, {judge}")


x_crit = np.array([v[x] for v in solve_ret])
y_crit = np.array([v[y] for v in solve_ret])
z_crit = np.array([f.subs({x:v[0], y:v[1]}) for v in zip(x_crit, y_crit)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, y1, z1, rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True)
ax.scatter(x_crit,y_crit, z_crit, c=['r','r','g','r'], s=50)
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.title("Polynomial Surface")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()