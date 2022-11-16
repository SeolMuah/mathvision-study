import numpy as np
import matplotlib.pyplot as plt 

sigma = 10
x0 = np.array([i for i in range(1, 100)])
y0 = -2*(x0-40)**2 + 30 + sigma*np.random.randn(len(x0))
x1 = np.random.rand(900)*100
y1 = np.random.rand(900)*8700-7170

x = np.concatenate([x0, x1], axis=0)
y = np.concatenate([y0, y1], axis=0)

A0 = np.array([x0**2, x0, np.ones(len(x0))]).T
A = np.array([x**2, x, np.ones(len(x))]).T


n_data = len(x)
N = 2000 #iteration
T = 2*sigma #residual threshold
n_sample = 3
max_cnt = 0
best_model = [.0, .0, .0]


for i in range(N) :

    k = np.floor(n_data*np.random.rand(n_sample)).astype(int)
    Ak = np.array([x[k]**2, x[k], np.ones(len(k))]).T
    pk = np.linalg.pinv(Ak) @ y[k]
    residual = np.abs(y-A @ pk)
    cnt = (residual < T).sum()

    if cnt > max_cnt :
        best_model = pk
        max_cnt = cnt

residual = np.abs(y-A @ best_model)
in_k = residual < T

A2 = np.array([x[in_k]**2, x[in_k], np.ones(in_k.sum())]).T
p = np.linalg.pinv(A2) @ y[in_k]

residual = np.abs(y-A @ p)
ink = residual < T
plt.title(f'Iter:{N}, Threshold:{T}, Outlier ratio:{100*np.invert(ink).sum()/len(x):.0f}%')
plt.scatter(x, y)
plt.scatter(x[ink], y[ink], color='red')
plt.plot(x0, A0 @ p, 'green')
plt.show()