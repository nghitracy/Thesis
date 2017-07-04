import numpy as np

p0 = np.array([[[123,456]]])
p1 = np.array([[7,8]])
print p0
for i,(x,y) in enumerate(zip(p0,p1)):
    x0,y0=x.ravel()
    x1,y1 = y.ravel()
print x0,y0,x1,y1


list=[]
list.append(0)
list.append(3)
list.append(39)
list.append(34)
list.append(7)
print list
list.sort()
print list



a =np.array([1,2])
print a
t,p = a.ravel()
print t,p

nghi=np.array([[99,33]])
nghi= np.float32(nghi)
nghi = np.array([nghi])
print nghi