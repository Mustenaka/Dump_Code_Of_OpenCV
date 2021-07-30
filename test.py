import cv2
import numpy as np
from matplotlib import pyplot as plt
 
# 插入上一段代码中定义的函数
 
 
 
 
 
calculator = cv2.imread('./data/image/calculator.tif')
calculator1 = O_R(1,calculator,np.ones((1,71)))
 
f,ax = plt.subplots(1,2,figsize=(15,15))
ax[0].imshow(calculator)
ax[1].imshow(calculator1)