# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:06:35 2020

@author: hernan
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import sim

#INTERSECCION RECTA-PLANO

#recta:  vt+vo

#plano : < n , r-r0 > = 0

#interseccion:  <n, (vt1+v0)- r0> =0








class Recta():
    
  def __init__(self, v, v0,t):
     self.v = v
     self.t= t
     self.v0=v0
     

     
  def actualizarV(self,val):
     self.v = val
     
  def actualizarV0(self,val1):
     self.v0 = val1
     
  def actualizart(self,val2):
     self.t = val2
     
     
     
  def r(self):
      
    '''devuelve el rayo  [Xi,Yi,Zi] de tamaño  3 x len(t)'''
      
    vVect = self.v.reshape(3,-1)
    tVect= self.t.reshape(1,-1)
    v0Vect= self.v0*np.ones((1,tVect.shape[1])).T

    rr=  np.dot(vVect,tVect)+v0Vect.T
     
    return  rr
      

                           
                           
                           

def interseccion(recta, plano):
    
  '''Esta función devuelve el punto i=(X,Y,Z) en el que la recta interseca al plano '''
    
  normal=(plano.normal) #normal del plano
  r0=np.array([0,0,plano.z0])   #punto del plano 
  v = recta.v   #v    direccion del rayo
  v0= recta.v0  #v0   posicion de la camara

  aux=(r0- v0)  
  t=( np.inner(normal,aux ) ) / ( np.inner(normal,v)) #interseccion
  i= np.dot(v,t)+v0
  
  return i
##############################################################################



  
  
  

hs5=10 

Vmax5=300
Vmin5=0

x5 = np.linspace(Vmin5, Vmax5, hs5)
y5 = np.linspace(Vmin5, Vmax5, hs5)
X5, Y5 = np.meshgrid(x5, y5)

t1=np.arange(-50,0)
t2=np.arange(-300,0)

normalPlano=np.array([-0.05, 0.025])#n
z0Plano=10# r0
p1=sim.Plano(normalPlano,[X5,Y5],z0Plano)




Drecta1=[0,0,1]#v
Drecta1=Drecta1/np.linalg.norm(Drecta1)


pPunto1=[200,100, 50] #v0  
recta1=Recta(Drecta1,pPunto1,t1)

Drecta2=[-5,-8,1]#v
Drecta2=Drecta2/np.linalg.norm(Drecta2)

pPunto2=[0,0,30] #v0   posicion de la camara FE
recta2=Recta(Drecta2,pPunto2,t2)


II1=interseccion(recta1,p1)
II2=interseccion(recta2,p1)

r1=recta1.r()
r2=recta2.r()

#GRAFICA

#%%





fig = plt.figure()
ax5 = fig.add_subplot(111,projection='3d')

ax5.scatter(X5, Y5, p1.z,marker='o',c='red',linewidths=3);
# ax5.plot_surface(X5, Y5, p1.z, color='red');
ax5.plot(r1[0],r1[1],r1[2], color='blue');
ax5.plot(r2[0],r2[1],r2[2], color='green');
ax5.scatter(pPunto2[0],pPunto2[1],pPunto2[2], color='black',linewidths=3) 
ax5.scatter(pPunto1[0],pPunto1[1],pPunto1[2], color='black',linewidths=3)   

ax5.scatter(II1[0],II1[1],II1[2],marker='o',c='blue',linewidths=15)
ax5.scatter(II2[0],II2[1],II2[2],marker='o',c='green',linewidths=15)
ax5.set_ylabel('Eje y')
ax5.set_xlabel('Eje x')
ax5.set_zlabel('Eje z')
print('----------------------')
print('Interseccion recta 1 (Azul) ') 
print(II1)
print('\n')
print('----------------------')
print('Interseccion recta 2 (Verde)')
print(II2)

print('\n')
print('----------------------')
print('Distancia [teniendo en cuenta (X,Y,Z) ]  entre puntos de interseccion',np.linalg.norm(II1 - II2, axis=0) )
print('\n')
print('Distancia [teniendo en cuenta (X,Y) ]  entre puntos de interseccion',np.linalg.norm(II1[0:2] - II2[0:2], axis=0) )
print('----------------------')