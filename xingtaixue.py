import cv2
import numpy as np
from matplotlib import pyplot as plt
 


DEBUG = False
 
#　测地膨胀
def D_g(n,f,b,g):
    if n==0:
        return f
    if n==1:
        if DEBUG:
            cv2.imshow('g',g)
            cv2.imshow('img',cv2.dilate(f,b,iterations=1))
            cv2.imshow('min',np.min((cv2.dilate(f,b,iterations=1),g),axis=0))
            #cv2.imshow('c',np.min((cv2.dilate(f,b,iterations=1),g),axis=0)-cv2.dilate(f,b,iterations=1))
            cv2.waitKey()
            cv2.destroyAllWindows()
            #from IPython.core.debugger import Tracer; Tracer()()
            #print((cv2.dilate(f,b,iterations=1)<=g).all())
        return np.min((cv2.dilate(f,b,iterations=1),g),axis=0)
    return D_g(1,D_g(n-1,f,b,g),b,g)
 
# 测地腐蚀
def E_g(n,f,b,g):
    if n==0:
        return f
    if n==1:
        return np.max((cv2.erode(f,b,iterations=1),g),axis=0)
    return E_g(1,E_g(n-1,f,b,g),b,g)
    
# 膨胀重建
def R_g_D(f,b,g):
    if DEBUG:
        cv2.imshow('origin',f)
        cv2.waitKey()
        #cv2.destroyAllWindows()
    img = f
    while True:
        new = D_g(1,img,b,g)
        cv2.destroyAllWindows()
        if (new==img).all():
            return img
        img = new
        
# 腐蚀重建
def R_g_E(f,b,g):
    img = f
    while True:
        new = E_g(1,img,b,g)
        if (new==img).all():
            return img
        img = new
 
# 重建开操作
def O_R(n,f,b,conn=np.ones((3,3))):
    erosion=cv2.erode(f,b,iterations=n)
    return R_g_D(erosion,conn,f)
 
# 重建闭操作
def C_R(n,f,b,conn=np.ones((3,3))):
    dilation = cv2.dilate(f,b,iterations = n)
    return R_g_E(dilation,conn,f)


calculator = cv2.imread('Image/img3.jpg')
calculator1 = O_R(1,calculator,np.ones((1,71)))
 
f,ax = plt.subplots(1,2,figsize=(15,15))
ax[0].imshow(calculator)
ax[1].imshow(calculator1)