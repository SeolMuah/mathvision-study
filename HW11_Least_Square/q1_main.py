import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=100)
#(1) Obtain the best binarized image of the provided sample image by a global thresholding
img = cv2.imread('./hw11_sample.png', cv2.IMREAD_GRAYSCALE)

# 130을 threshold로 하는 이미지
_, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
# otsu algorithm을 적용한 이미지
t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

imgs = {'Original': img, 't:130':t_130, f'otsu:{t:.0f}': t_otsu}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3 , i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()

#(2) Approximate the background of the sample image by a 2 nd order polynomial surface and then display it as an image
#I(x,y) = ax^2 + by^2 + cxy + dx + ey + f
height, width = img.shape
xs = np.array([i for i in range(width)])
ys = np.array([i for i in range(height)])

X,Y = np.meshgrid(xs,ys, indexing='xy')
X.shape #(353, 438)
Y.shape #(353, 438)
pos = np.dstack((X,Y)) #(353, 438, 2)
pos = pos.reshape(-1, 2) #(154614, 2)
A = []
I = []

mean_x = pos[:,0].mean()
std_x = pos[:,0].std()
mean_y = pos[:,1].mean()
std_y = pos[:,1].std()
for x,y in pos :
    #좌표 Z표준화
    # norm_x = (x-mean_x)/std_x
    # norm_y = (y-mean_y)/std_y
    norm_x = x
    norm_y = y

    A.append((norm_x**2, norm_y**2, norm_x*norm_y, norm_x, norm_y, 1))
    I.append(img[y,x])

A = np.array(A)
I = np.array(I)
A.shape, I.shape #((154614, 6), (154614,))



A_plus = np.linalg.inv((A.T @ A)) @ A.T
A_plus = np.linalg.pinv(A)
P = A_plus @ I
pos.shape #(154614, 2)

#근사 곡면 시각화
fig = plt.figure()
# plt.title("polynomial Surface")
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, (A @ P).reshape(height, width), rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True) 
# surf = ax.plot_surface(X, Y, I.reshape(height, width), rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True) 
fig.colorbar(surf,shrink=0.5,aspect=5)
# plt.tight_layout()
plt.title("Polynomial Surface")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

background = (A @ P).reshape(height, width).astype(np.uint8)
plt.imshow(background, cmap='gray')
plt.show()

#3. Subtract the approximated background image from the original and binarize the result (background-subtracted image) to obtain the final best binarized
sub_img = img.astype(np.int32) - background.astype(np.int32)

np.min(sub_img) #-85
np.max(sub_img) # 15
sub_img += np.min(sub_img)

t, t_otsu = cv2.threshold(sub_img.astype(np.uint8), -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.imshow(t_otsu, cmap='gray')
plt.show()