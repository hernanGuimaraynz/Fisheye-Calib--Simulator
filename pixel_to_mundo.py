# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:31:03 2020

@author: hernan
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
# import camara05 as  camara2
import  camara2
import sim

np.set_printoptions(suppress=True)



#%DEFINO UNA CAMARA FISHEYE (con parametros reales) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xfeo=50; # coord. X de FE en trama {m=0} medida en [m]
yfeo=50 # coord. Y de FE en trama {m=0} medida en [m]
zfeo=-15;# coord. Z de FE en trama {m=0} medida en [m]25

#Angulos de Euler ZYX para la orientacion de la camara FE en trama {0}

alfa_fe=0*np.pi/180
beta_fe=0*np.pi/180
gama_fe=0*np.pi/180

FE=camara2.Camara(1, xo=xfeo, yo=yfeo, zo=zfeo, alpha0= alfa_fe, beta0= beta_fe, gamma0= gama_fe)
# FE.camera_info()

kfe=FE.k

###############DEFINO LA INCERTEZA  EN LOS PARAMETROS DE LA CAMARA FE########################


FlagRuidoPARAM=1 # Flag de incerteza en los parametros FE

##################################################################################


powAngulo=(5*np.pi/180) # 5° de incerteza en los angulos Euler de la cámara
powPosicion=1         # 1 metros de incerteza en la posicion de la cámara X,Y,Z
powK=5*0.01 # incerteza de K , indica un porcentaje del valor "real" de k.

##################################################################################

# sigma * np.random.randn(...) + mu

ruidoa1FE=powAngulo*np.random.randn()
ruidoa2FE=powAngulo*np.random.randn()
ruidoa3FE=powAngulo*np.random.randn()
ruidopxFE=powPosicion*np.random.randn()
ruidopyFE=powPosicion*np.random.randn()
ruidozFE=(powPosicion)*np.random.randn()
ruidoKFE=(kfe*powK)*np.random.randn()


alfa_feR =alfa_fe+ruidoa1FE*FlagRuidoPARAM
beta_feR=beta_fe+ruidoa2FE *FlagRuidoPARAM
gama_feR=gama_fe+ruidoa3FE*FlagRuidoPARAM
xfeoR=xfeo+ruidopxFE*FlagRuidoPARAM
yfeoR=yfeo+ruidopyFE*FlagRuidoPARAM
zfeoR=zfeo+ruidozFE*FlagRuidoPARAM
kfe_R=kfe +ruidoKFE*FlagRuidoPARAM

#vector de parametros reales
paramFE=[xfeo,yfeo,zfeo,alfa_fe,beta_fe,gama_fe]
#vector de parametros ruidosos
paramFE_R=[xfeoR,yfeoR,zfeoR,alfa_feR,beta_feR,gama_feR]

paramFE_R_INICIAL=paramFE_R
####################################################################################################################################

##############          Defino un objeto camara FE con parametros ruidosos.           ##############################################

FE_R=camara2.Camara(2, xo=xfeoR, yo=yfeoR, zo=zfeoR, alpha0= alfa_feR, beta0= beta_feR, gamma0= gama_feR,k0=kfe_R)
####################################################################################################################################

#%%
#DEFINICION DE CLASES Y FUNCIONES

class Recta():
    

  def __init__(self, vv,t=0):
     self.v = vv[:,1]
     self.t= t
     self.v0=vv[:,0]
     
  def actualizarV(self,val):
     self.v = val
     
  def actualizarV0(self,val1):
     self.v0 = val1
     
  def actualizarT(self,val2):
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
##############################################################




def onMouse(event, x, y, flags, params):
 ''' 
 Esta funcion permite:
 Clickear en la imagen FE y obtener las coordenadas (u,v) del pixel marcado (a partir de el punto marcado se calculan las 4 esquinas).
 Despues:
 __Calcula el rayo del pixel, es decir el versor direccion a partir de la posicion de la camara FE (con la camara ruidosa).
 __Calcula interseccion del rayo con el plano en coordendas del mundo en Zm=0  y grafica las 4 esquinas resultantes.
 __Mide la dispersion de las esquinas del pixel en metros.

 '''  
 #cuando se clickea en la imagen
 if event is cv2.EVENT_LBUTTONDOWN:
  plt.close()
  global data
  #gurado las coordenadas (u,v)
  data=[x, y]

  
  cv2.imshow('Imagen FE: Clickee un pixel para ver su retroproyecion',(image)) 
   
  
    #reescaleo (u,v) al tamaño original de la imagen para las 4 esquinas
    
   #punto original( esquina inferior izq)
  dd=data*(1920/L*np.ones_like(data))
  #esquina superior izq
  dd1=dd+[0 ,FE.tampixfe]
  #esquina superior derecha
  dd2=dd+[FE.tampixfe ,FE.tampixfe]
  #esquina inferior derecha
  dd3=dd+[FE.tampixfe ,0]

 #calculo los rayos de los pixeles a partir de la posicion de la camara FE_R
 
  vm=FE_R.CameraToWorld_Proyect(dd)
  vm1=FE_R.CameraToWorld_Proyect(dd1)
  vm2=FE_R.CameraToWorld_Proyect(dd2)
  vm3=FE_R.CameraToWorld_Proyect(dd3)

  r0=Recta(vm)
  r1=Recta(vm1) 
  r2=Recta(vm2)  
  r3=Recta(vm3)
  

   #calculo las intersecciones de los rayos con el plano Zm=0 
  i_0=interseccion(r0,p1)
  i_1=interseccion(r1,p1)
  i_2=interseccion(r2,p1)
  i_3=interseccion(r3,p1)
  
  
  
  #GRAFICO#################################################
  
  ddd1=[dd[0],dd1[0],dd2[0],dd3[0],dd[0]]
  ddd2=[dd[1],dd1[1],dd2[1],dd3[1],dd[1]] 
  
  vmm1=[i_0[0],i_1[0],i_2[0],i_3[0],i_0[0]]
  vmm2=[i_0[1],i_1[1],i_2[1],i_3[1],i_0[1]]
  
  Dist_y=np.linalg.norm(i_0-i_1,axis=0)

  Dist_x=np.linalg.norm(i_0-i_3,axis=0)
    
  fig =plt.figure()

  ax = fig.add_subplot(121)
  ax1 = fig.add_subplot(122)

  ax.plot(dd[0],dd[1],'bo')
  ax.plot(dd1[0],dd1[1],'ro')
  ax.plot(dd2[0],dd2[1],'go')
  ax.plot(dd3[0],dd3[1],'ko')
  
  ax.plot(ddd1,ddd2,'m')
  
  ax.set_xlim([0,1920])
  ax.set_ylim([0,1920])
  ax.set_title('Pixel marcado: ' + str(dd))
  ax.set_xlabel('Imagen FE.')

  ax1.plot(vmm1,vmm2,'b')
  ax1.set_title('Dispersion del pixel: X:'+ '({0:.3f})'.format(Dist_x)+' metros' + ' Y:'+ '({0:.3f})'.format(Dist_y) +' metros' )
  ax1.set_xlabel('Plano Mundo Zm= 0.')
 
  # Dist_pix=np.linalg.norm(dd1-dd3,axis=0)
  print('\n')
  print('-----------------')
  print('Pixel marcado: ',dd)
  print('Dispersion en metros del pixel en x : ',Dist_x,' metros')
  print('Dispersion en metros del pixel en y : ',Dist_y,' metros')
  print('Dispersion en metros del pixel Total : ',(Dist_y+Dist_x) ,' metros') 
  print('-----------------')
 
  
#%%

#creo la imagen de la camara FE donde voy a marcar los pixeles (La armé a la mitad de resolucion de la original que es 1920 x 1920
#para poder clickear en la imagen mas facil)

global L


L=960    #1920 #(tamaño original)

image=(np.ones((L,L)))


image[int(L/2),:]=0
image[:,int(L/2)]=0

image[int(L/4),:]=0
image[:,int(L/4)]=0

image[int(3*L/4),:]=0
image[:,int(3*L/4)]=0


#### ## ## ## ## ## ## ## ## ## 

#Defino el plazo Zm=0 para calcular la interseccion
n=[0,0]
normal=[0,0,1]
z0=0
p1=sim.Plano(n,normal,z0)


## ## ## ## ## ## ## ## ## ## 



plt.rcParams["figure.figsize"] = (10, 6) 
  
  
 #MUESTRO LA IMAGEN
cv2.imshow('Imagen FE: Clickee un pixel para ver su retroproyecion',image) 
 
#CORRE RUTINA DEL MOUSE 
cv2.setMouseCallback('Imagen FE: Clickee un pixel para ver su retroproyecion',onMouse) 



























