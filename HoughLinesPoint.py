import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean
 
 
def nothing(x):  # 滑动条的回调函数
    pass
 
 
imagePath = "Image/img7.png"
zoom = 1

source = cv2.imread(imagePath)
print("原图shape:",source.shape)
src = cv2.resize(source,(int(source.shape[1] * zoom), int(source.shape[0] * zoom)))
print("缩放后shape:",src.shape)

h,w,_ = src.shape
print(src.shape)
cent_x = int(h / 2)
cent_y = int(w / 2)
cut_zoom = 10
cropped =  src[cent_x-cut_zoom:cent_x+cut_zoom, cent_y-cut_zoom:cent_y+cut_zoom]


print(cropped.shape)
cro_h,cro_w,_ = cropped.shape

#np.max()

max_pix = [0,0,0]
min_pix = [255,255,255]
mean_pix = [0,0,0]
for i in range(cro_h):
    for j in range(cro_w):
        # max
        li = cropped[i,j].tolist()
        for k in range(len(li)):
            if li[k] > max_pix[k]:
                max_pix[k] = li[k]
        # min
        for k in range(len(li)):
            if li[k] < min_pix[k]:
                min_pix[k] = li[k]
        # mean
        for k in range(len(li)):
            mean_pix[k] += li[k]
            

for k in range(len(mean_pix)):
    mean_pix[k] /= (cro_h * cro_w)
        
print("cropped pixiv is ",cropped[int(cro_h/2),int(cro_w/2)])
print("max pixiv is:",max_pix)
print("min pixiv is:",min_pix)
print("mean pixiv is:",mean_pix)

#cv2.imshow("cropped",cropped)
#cv2.imwrite("cropped.png",cropped)
#cv2.waitKey(0)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
mask = np.zeros([h+2, w+2],np.uint8)
""" cv2.floodFill(src,
              mask,
              (cent_x,cent_y),
              (0,0,0),
              (min_pix[0],min_pix[1],min_pix[2]),
              (max_pix[0],max_pix[1],max_pix[2])) """

print(cent_x,cent_y)
cv2.floodFill(src,
              mask,
              (cent_x,cent_y),
              (0,255,0),
              (255,255,255),
              (0,0,0),
               cv2.FLOODFILL_FIXED_RANGE)
              #cv2.FLOODFILL_FIXED_RANGE)

retval, dst = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# 高斯滤波, 中值滤波，卷积核用(3,3)
srcBlur = cv2.GaussianBlur(dst, (3, 3), 0)

cv2.imwrite("newimg.png",src)
#cv2.imwrite("newimg.png",newimg)
cv2.imshow("newimg",src)
cv2.waitKey(0)
# 灰度
#gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 二值化
#retval, dst = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

"""

# 边缘查找算法
edges = cv2.Canny(dst, 50, 150, apertureSize=3)
cv2.imwrite("Temp/edge1.jpg",edges)

WindowName = 'Approx'  # 窗口名
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口
 
cv2.createTrackbar('threshold', WindowName, 0, 100, nothing)  # 创建滑动条
cv2.createTrackbar('minLineLength', WindowName, 0, 50, nothing)  # 创建滑动条
cv2.createTrackbar('maxLineGap', WindowName, 0, 100, nothing)  # 创建滑动条
 
"""
"""
HoughLinesP 参数：
src：输入图像，必须8-bit的灰度图像
rho：生成极坐标时候的像素扫描步长
theta：生成极坐标时候的角度步长
threshold：阈值，只有获得足够交点的极坐标点才被看成是直线
lines：输出的极坐标来表示直线
minLineLength：最小直线长度，比这个短的线都会被忽略。
maxLineGap：最大间隔，如果小于此值，这两条直线 就被看成是一条直线。
"""
"""

while(1):
    img = src.copy()
    threshold = cv2.getTrackbarPos('threshold', WindowName)  # 获取滑动条值
    minLineLength = 2 * cv2.getTrackbarPos('minLineLength', WindowName)  # 获取滑动条值
    maxLineGap = cv2.getTrackbarPos('maxLineGap', WindowName)  # 获取滑动条值

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)
    # print(lines)


    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        

    cv2.imshow(WindowName, img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()


"""