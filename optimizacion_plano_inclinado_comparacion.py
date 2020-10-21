# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:22:14 2020

@author: hernan
"""



import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
np.set_printoptions(suppress=True)
import os
import sim
import camara2



#%DEFINO UNA CAMARA FISHEYE (con parametros reales) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xfeo=50; # coord. X de FE en trama {m=0} medida en [m]
yfeo=50 # coord. Y de FE en trama {m=0} medida en [m]
zfeo=-15;# coord. Z de FE en trama {m=0} medida en [m]25

#Angulos de Euler ZYX para la orientacion de la camara FE en trama {0}
alfa_fe=0#*np.pi/180
beta_fe=0#*np.pi/180
gama_fe=0#*np.pi/180

FE=camara2.Camara(1, xo=xfeo, yo=yfeo, zo=zfeo, alpha0= alfa_fe, beta0= beta_fe, gamma0= gama_fe)
# FE.camera_info()

kfe=FE.k

###############DEFINO LA INCERTEZA  EN LOS PARAMETROS DE LA CAMARA FE########################


FlagRuido=1# Flag de ruido de calibracion en la imagen [1 pixel]

FlagRuidoPARAM=1 # Flag de incerteza en los parametros FE


##################################################################################


powAngulo=5#*np.pi/180 # 5° de incerteza en los angulos Euler de la cámara
powPosicion=1         # 1 metros de incerteza en la posicion de la cámara X,Y,Z
powK=(5*0.01) # incerteza de K , indica un porcentaje del valor "real" de k.

##################################################################################



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

paramFE_R_INICIAL=np.array(paramFE_R)
####################################################################################################################################

##############          Defino un objeto camara FE con parametros ruidosos.           ##############################################

FE_R=camara2.Camara(2, xo=xfeoR, yo=yfeoR, zo=zfeoR, alpha0= alfa_feR, beta0= beta_feR, gamma0= gama_feR,k0=kfe_R)


###############################################################################################################################################
#DEFINO LA FUNCION DE COSTO , CUYA MINIMIZACION PERMITE OPTIMIZAR LOS PARAMETROS DEL PLANO INICIAL QUE AJUSTA A LA SUPERFICIE REAL Y A LOS PARAMETROS DE LA CAMARA FE 
###############################################################################################################################################

#mejor el 3 o el 8
def costo(pars, rayo_pix_costo   ):
  # pars : [n1,n2,z0,xfeR,yfeR,zfeR,alfaR,betaR,gamaR] PARAMETROS A OPTIMIZAR
  # rayo_pix_costo : pixeles en la imagen FE de los puntos de calibracion

  # defino la camara cuyos parametros se van optimizando
  FE_Rcosto.update_parametros([pars[3], pars[4], pars[5], pars[6], pars[7], pars[8]])

  p1.update_normal([pars[0],pars[1],0.9])
  p1.actualizarZ0(pars[2])

  #retroproyecto los rayos al mundo
  V_m_costo=FE_Rcosto.CameraToWorld_Proyect(rayo_pix_costo)

  # interseco los rayos con el plano a optimizar
  r1_costo=sim.interseccion(V_m_costo,p1)

  #error
  aux=r1_costo-r2
  
  DistanciaTotal_costo=np.sum(np.linalg.norm(aux[0:2],axis=0))


  return DistanciaTotal_costo


def costo2(pars, rayo_pix_costo   ):
  # pars : [n1,n2,z0,xfeR,yfeR,zfeR,alfaR,betaR,gamaR] PARAMETROS A OPTIMIZAR
  # rayo_pix_costo : pixeles en la imagen FE de los puntos de calibracion

  # defino la camara cuyos parametros se van optimizando
  FE_Rcosto.update_parametros([pars[3], pars[4], pars[5], pars[6], pars[7], pars[8]])

  p1.update_normal([pars[0],pars[1],0.9])
  p1.actualizarZ0(pars[2])

  #retroproyecto los rayos al mundo
  V_m_costo=FE_Rcosto.CameraToWorld_Proyect(rayo_pix_costo)

  # interseco los rayos con el plano a optimizar
  r1_costo=sim.interseccion(V_m_costo,p1)

  #error
  rayo_pix_costo1=FE.WorldToCamera_Proyect(r1_costo)
  DistanciaPix=np.sum(np.linalg.norm(rayo_pix_costo-rayo_pix_costo1,axis=1))

  return DistanciaPix

alfaPix=1/100

def costo3(pars, rayo_pix_costo   ):
  # pars : [n1,n2,z0,xfeR,yfeR,zfeR,alfaR,betaR,gamaR] PARAMETROS A OPTIMIZAR
  # rayo_pix_costo : pixeles en la imagen FE de los puntos de calibracion

  # defino la camara cuyos parametros se van optimizando
  FE_Rcosto.update_parametros([pars[3], pars[4], pars[5], pars[6], pars[7], pars[8]])

  p1.update_normal([pars[0],pars[1],0.9])
  p1.actualizarZ0(pars[2])

  #retroproyecto los rayos al mundo
  V_m_costo=FE_Rcosto.CameraToWorld_Proyect(rayo_pix_costo)

  # interseco los rayos con el plano a optimizar
  r1_costo=sim.interseccion(V_m_costo,p1)

  #error
  rayo_pix_costo1=FE.WorldToCamera_Proyect(r1_costo)
  DistanciaPix=np.sum(np.linalg.norm(rayo_pix_costo-rayo_pix_costo1,axis=1))
  
  aux=r1_costo-r2
  
  DistanciaTotal_costo=np.sum(np.linalg.norm(aux[0:2],axis=0))


  return DistanciaPix*alfaPix + DistanciaTotal_costo


#######################################################################################################################
#      Defino la superficie 'real' y un plano cuyos parametros van a ser optimizados para ajustar la superficie
#######################################################################################################################

######################################################
#Parametros iniciales del plano y la superficie "real"
######################################################
#plano a optimizar
normalPlano=np.array([0, -0.005,0.9])
z0Plano=-1

#superficie real 
a=0.0094
mu=0.73
h=1

######################################################

#Defino la cantidad de puntos a utilizar en la calibracion


# print('--------------------------------------------------')
# print('              OPTIMIZACION')
# print('--------------------------------------------------')

# print('Nro Puntos de calibracion: ' + str(nroCalibracion))
# print('\n')
 
nroCalibracionMin=10
nroCalibracionMax=20

nroCal=np.arange(nroCalibracionMin,nroCalibracionMax+1)
# m=nroCalibracion
# VMIN=0
# VMAX=100

#Defino el nro de  iteraciones
iteraciones=20

tol = 1e-7
tol2 = 0.001

options={'maxiter' : 20000}

#estos son todos los metodos que tiene cargados el optimize.minimize
metodos=['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP','trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']

#lista de los metodos que voy a probar
imetVect=[0,1,2,3,5,6,7,8,9] 

# imetVect=[0] 
#genero los puntos a partir del generador rs , partiendo de una semilla, la cual se guarda en el disco
semilla=np.random.seed()

rs=np.random.RandomState(semilla)

# Genero una matriz de  numeros semialeatorios que van de VMIN a VMAX
muB=50
sigmaB=10


VXY_V=[]
Vpars=[]
Vcosto=[]
VmapError=[]
VimgError=[]
VmapErrorVal=[]
VimgErrorVal=[]
vE_map3D=[]
vE_map3Dval=[]




vvvv2d0=[]
vvvvimg0=[]
vvvv2d1=[]
vvvvimg1=[]
VparamErr0=[]
VparamErr1=[]

vvvv2d2=[]
vvvvimg2=[]
VparamErr2=[]





##########################################################################################################
                    #OPTIMIZACION / MINIMIZACION DEL ERROR 
#########################################################################################################

V2d0=[]
Vimg0=[]
VVparamErr0=[]

V2d1=[]
Vimg1=[]
VVparamErr1=[]

V2d2=[]
Vimg2=[]
VVparamErr2=[]

#%%

for cal in nroCal:
    

 nroCalibracion=cal
 m=nroCalibracion
 print('Nro de puntos calibracion: '+str(nroCalibracion) +'/'+str(nroCalibracionMax)  )

 xy=rs.normal(muB, sigmaB, (nroCalibracion,2))

 X=xy[:,0]
 Y=xy[:,1]

 # print(  [X,Y]   )
 #altura de la superficie real
 Z=-sim.f2(X,Y,a,mu,h)

 #genero el plano inicial
 p1=sim.Plano(normalPlano,[X,Y],z0Plano)

 pars= np.hstack(([p1.normal[0],p1.normal[1],p1.z0] ,paramFE_R))

 #defino la camara cuyos parametros van a ir siendo optimizados
 FE_Rcosto=camara2.Camara(2, xo=pars[3], yo=pars[4], zo=pars[5], alpha0=pars[6], beta0=pars[7], gamma0=pars[8])

 FE_Opt=camara2.Camara(2, xo=pars[3], yo=pars[4], zo=pars[5], alpha0=pars[6], beta0=pars[7], gamma0=pars[8],k0=FE.k)
 #puntos de calibracion 
 r2=np.array([X,Y,Z])

 #proyecto los puntos a rayos en coordenadas de la camara
 rayo_pix=FE.WorldToCamera_Proyect(r2)

 # np.save('semilla_planoInclinado',semilla)
 # np.save('RandomState_planoInclinado',rs)

 Vpars=[]
 VmapError=[]
 VimgError=[]
 VmapErrorVal=[]
 VimgErrorVal=[]
 vE_map3D=[]
 vE_map3Dval=[]
 Xvector=np.zeros((iteraciones, m))
 Yvector=np.zeros((iteraciones, m))
 XvectorVal=np.zeros((iteraciones, 1))
 YvectorVal=np.zeros((iteraciones, 1))  
 vvvv2d0=[]
 vvvvimg0=[]
 vvvv2d1=[]
 vvvvimg1=[]
 VparamErr0=[]
 VparamErr1=[]
    

 for energia in range(3):

  for i in range(len(imetVect)): #Recorro los metodos de minimizacion 


   


   for k in range(iteraciones):# Se repite el proceso para analizar la estabilidad del metodo
    


 ####Defino la incerteza en la calibracion de las posiciones (u,v) en pixeles de la imagen FE (Para los puntos de calibracion y de validacion )  ################


    if FlagRuido==1:

       ruido_Pixel_Calibracion = np.round( (2*np.random.rand(nroCalibracion, 2))  -1)   #  [-1 a 1] error de pixel
       ruido_Pixel_Validacion =  np.round((2*np.random.rand(1,2))  -1) 

    else:
        ruido_Pixel_Validacion =0
        ruido_Pixel_Calibracion =0
############################################################################################################

    rayo_pixi= rayo_pix + ruido_Pixel_Calibracion


   #defino el metodo de minimizacion
    imet =imetVect[i]

    nit=0
    F=0
#costo inicial del proceso
    if energia == 0 :
       F0=costo(pars, rayo_pixi ) #COSTO INICIAL
    if energia == 1 :
       F0=costo2(pars, rayo_pixi ) #COSTO INICIAL

    if energia == 2 :
       F0=costo3(pars, rayo_pixi ) #COSTO INICIAL
    FF=True
    OPTvector=[]


##### INICIO  del bucle  de minimizacion ######################################
    while (FF):

     if energia == 0 :  
        ret1 = minimize(costo, pars, args=(rayo_pixi), method=metodos[imet], tol=tol)
     if energia == 1 :  
        ret1 = minimize(costo2, pars, args=(rayo_pixi), method=metodos[imet], tol=tol)
     if energia == 2 :  
        ret1 = minimize(costo3, pars, args=(rayo_pixi), method=metodos[imet], tol=tol)

     F=ret1.fun
 
 
     if F< tol :
     
      # print('------------------------------')
      # print('Tolerancia de error alcanzada') 
      # print('------------------------------')     
       break 
 
     if (str(F)=='nan'):
     
      # print('------------------------------')
      # print('F no es un numero valido') 
      # print('------------------------------')
       break

     if F>=F0 or abs(F-F0)<=tol2:
     
      # print('------------------------------')     
      # print('Optimizacion Finalizada')  
      # print('------------------------------')
       break     
     else:

       OPTvector.append([metodos[imet],ret1.fun,ret1.x])
     
       F0=F  #actualizo la opt
       pars2=ret1.x #actualizo PARS
      
     nit+=1
     if nit >=5:
       imet=0 
 
################ FIN  del bucle de minimizacion ###########################


#obtengo los parametros optimizados
    parsOpt=ret1.x
    Vpars.append(parsOpt)
   
  
    Vcosto.append([costo(parsOpt,rayo_pix)])
  
     
 ########################################### ########################################### ###########################################
    ##########R ealizo las proyecciones y retroproyecciones con los parametros optimizados ##########
 ########################################### ########################################### ###########################################
    
    
################     CALIBRACION     ##########################################


#actualizo la Camara con los parametros optimizados

    FE_Opt.update_parametros([parsOpt[3], parsOpt[4], parsOpt[5], parsOpt[6], parsOpt[7], parsOpt[8]])
   
    vectParamFinal=np.hstack((parsOpt[0:2],parsOpt[2],parsOpt[3:9]))

#genero el plano optimizado
    normalPlanoOpt=np.array([parsOpt[0] ,parsOpt[1],0.9])
   
    p1opt=sim.Plano(normalPlanoOpt,[X,Y],parsOpt[2])



   #retroproyecto los rayos al mundo
    V_mF=FE_Opt.CameraToWorld_Proyect(rayo_pixi)


   # interseco los rayos con el plano a optimizar
    r1opt=sim.interseccion(V_mF,p1opt)

 
    rayo1_pix_OPT=FE.WorldToCamera_Proyect(r1opt)  

    '''Mido los diferentes tipos de error:
       Error 3D: Error 3D [en metros] entrre los puntos retroroyectados p=(X,Y,Z)  del plano optimizado (con  param FE optimizados)  y de la superficie real
   
       Error mapa :Error 2D [en metros] entrre los puntos retroroyectados al mapa satelital  del plano optimizado (con  param FE optimizados)  y de la superficie real
       
       Error imagen FE: Error [en pixeles] de la proyeccion en el plano imagen FE entre el plano optimizado (con  param FE optimizados)  y la superficie real

    '''

    pp=rayo_pix
    pp1=rayo1_pix_OPT

    #Error en mapa satelital (calibracion)
    DistanciaTotalMapa=np.mean(np.linalg.norm(r1opt[0:2]-r2[0:2],axis=0))

    #Error 3D (calibracion)
    DistanciaTotal3D=np.mean(np.linalg.norm(r1opt-r2,axis=0))

    #Error en imagen FE(calibracion)
    DistanciaTotalImg=np.mean(np.linalg.norm(pp1-pp,axis=1) )

    VimgError.append( DistanciaTotalImg ) 
    VmapError.append(DistanciaTotalMapa)
    vE_map3D.append(DistanciaTotal3D )
       
#Error en los parametros de la camara FE
    E_PARAM=abs(np.mean(Vpars,axis=0)[3:9]-paramFE)
    E_PARAM_I=abs(paramFE_R_INICIAL-paramFE)


   if energia ==0:

       vvvv2d0.append([np.mean(VmapError),metodos[imet]])  
       vvvvimg0.append([np.mean(VimgError),metodos[imet]]) 
     
       VparamErr0.append([E_PARAM,metodos[imet]])
       
   if energia ==1:
        
       vvvv2d1.append([np.mean(VmapError),metodos[imet]])  
       vvvvimg1.append([np.mean(VimgError),metodos[imet]])  
       VparamErr1.append([E_PARAM,metodos[imet]])
       
   if energia ==2:
        
       vvvv2d2.append([np.mean(VmapError),metodos[imet]])  
       vvvvimg2.append([np.mean(VimgError),metodos[imet]])  
       VparamErr2.append([E_PARAM,metodos[imet]])


 V2d0.append([vvvv2d0,cal])  
 Vimg0.append([vvvvimg0,cal])  
 VVparamErr0.append([VparamErr0,cal])

 V2d1.append([vvvv2d1,cal])  
 Vimg1.append([vvvvimg1,cal])  
 VVparamErr1.append([VparamErr1,cal])

 V2d2.append([vvvv2d2,cal])  
 Vimg2.append([vvvvimg2,cal])  
 VVparamErr2.append([VparamErr2,cal])

V2d1=np.array(V2d1)
Vimg1=np.array(Vimg1)
VVparamErr1=np.array(VVparamErr1)

V2d0=np.array(V2d0)
Vimg0=np.array(Vimg0)
VVparamErr0=np.array(VVparamErr0)

V2d2=np.array(V2d2)
Vimg2=np.array(Vimg2)
VVparamErr2=np.array(VVparamErr2)



np.save('Plano_inclinado_Error_paramFE_'+ '_' + 'ajuste2D',VVparamErr0)
np.save('Plano_inclinado_Error_XY_mapa_calibracion_'+ '_' + 'ajuste2D',V2d0)
np.save('Plano_inclinado_Error_img_calibracion_'+ '_' + 'ajuste2D',Vimg0)

np.save('Plano_inclinado_Error_paramFE_'+ '_' + 'ajusteImg',VVparamErr1)
np.save('Plano_inclinado_Error_XY_mapa_calibracion_'+ '_' +'ajusteImg',V2d1)
np.save('Plano_inclinado_Error_img_calibracion_'+ '_' + 'ajusteImg',Vimg1)

np.save('Plano_inclinado_Error_paramFE_'+ '_' + 'ajusteH',VVparamErr2)
np.save('Plano_inclinado_Error_XY_mapa_calibracion_'+ '_' +'ajusteH',V2d2)
np.save('Plano_inclinado_Error_img_calibracion_'+ '_' + 'ajusteH',Vimg2)



#%%

VVparamErr0=np.load('Plano_inclinado_Error_paramFE_'+ '_' + 'ajuste2D.npy')
VVparamErr1=np.load('Plano_inclinado_Error_paramFE_'+ '_' + 'ajusteImg.npy')
VVparamErr2=np.load('Plano_inclinado_Error_paramFE_'+ '_' + 'ajusteH.npy')


V2d0=np.load('Plano_inclinado_Error_XY_mapa_calibracion_'+ '_' + 'ajuste2D.npy')
V2d1=np.load('Plano_inclinado_Error_XY_mapa_calibracion_'+ '_' + 'ajusteImg.npy')
V2d2=np.load('Plano_inclinado_Error_XY_mapa_calibracion_'+ '_' + 'ajusteH.npy')

Vimg0=np.load('Plano_inclinado_Error_img_calibracion_'+ '_' + 'ajuste2D.npy')
Vimg1=np.load('Plano_inclinado_Error_img_calibracion_'+ '_' + 'ajusteImg.npy')
Vimg2=np.load('Plano_inclinado_Error_img_calibracion_'+ '_' + 'ajusteH.npy')


# VV2d0=[]
# VV2d1=[]
# VV2d2=[]

# VVimg1=[]
# VVimg0=[]
# VVimg2=[]


def changeForm(vect):
 hh0=[]  
 
 for k in range(len(nroCal)):  #recorre los punto de calibracion

  for i in range(len(imetVect)):#recorre los metodos
      
      
   hh0.append(    (vect[k][0][i])[0]  )
    
 hh0=np.array(hh0)
 hh0=hh0.reshape(-1,len(imetVect)) 
    
 return hh0
    
def changeForm2(vect,b):
 hh0=[]  
 
 for k in range(len(nroCal)):  #recorre los punto de calibracion

  for i in range(len(imetVect)):#recorre los metodos
      
      
   hh0.append( (vect[k][0][i])[0] )
    
 hh0=np.array(hh0)
 hh0=hh0[:,b].reshape(-1,len(imetVect))
    
 return hh0


VV2d0=changeForm(V2d0)
VV2d1=changeForm(V2d1)
VV2d2=changeForm(V2d2)

VVimg1=changeForm(Vimg1)
VVimg0=changeForm(Vimg0)
VVimg2=changeForm(Vimg2)

VVVparamErr0x=changeForm2(VVparamErr0,0)
VVVparamErr0y=changeForm2(VVparamErr0,1)
VVVparamErr0z=changeForm2(VVparamErr0,2)
VVVparamErr0a=changeForm2(VVparamErr0,3)
VVVparamErr0b=changeForm2(VVparamErr0,4)
VVVparamErr0g=changeForm2(VVparamErr0,5)

VVVparamErr1x=changeForm2(VVparamErr1,0)
VVVparamErr1y=changeForm2(VVparamErr1,1)
VVVparamErr1z=changeForm2(VVparamErr1,2)
VVVparamErr1a=changeForm2(VVparamErr1,3)
VVVparamErr1b=changeForm2(VVparamErr1,4)
VVVparamErr1g=changeForm2(VVparamErr1,5)

VVVparamErr2x=changeForm2(VVparamErr2,0)
VVVparamErr2y=changeForm2(VVparamErr2,1)
VVVparamErr2z=changeForm2(VVparamErr2,2)
VVVparamErr2a=changeForm2(VVparamErr2,3)
VVVparamErr2b=changeForm2(VVparamErr2,4)
VVVparamErr2g=changeForm2(VVparamErr2,5)


for i in range(len(imetVect)):

   fig=plt.figure(figsize=(10,10))

   ax1 = fig.add_subplot(211)
   ax2 = fig.add_subplot(212)

   
   fig2=plt.figure(figsize=(10,10))

   ax10 = fig2.add_subplot(231)
   ax20 = fig2.add_subplot(232)
   ax3 = fig2.add_subplot(233)
   ax4 = fig2.add_subplot(234)
   ax5 = fig2.add_subplot(235)
   ax6 = fig2.add_subplot(236)
   
   
   
   
   ax1.plot(nroCal,VV2d0[:,i],'b-x',label='Ajuste en el mapa',ms=5)
   # ax1.plot(nroCal,VV2d1[:,i],'r-o',label='Ajuste en la imagen')
   ax1.plot(nroCal,VV2d2[:,i],'g-x',label='Ajuste 2D + imagen')
   
   ax2.plot(nroCal,VVimg0[:,i],'b-x',label='Ajuste en el mapa',ms=5)
   # ax2.plot(nroCal,VVimg1[:,i],'r-o',label='Ajuste en la imagen')
   ax2.plot(nroCal,VVimg2[:,i],'g-x',label='Ajuste 2D + imagen')

   
   
   ax1.set_xlabel('Nro Puntos de Calibración')
   ax1.set_ylabel('Error [metros]')
   ax1.set_title('Metodo: '+str( metodos[imetVect[i]] ) +' ///  Error 2D')

   ax2.set_xlabel('Nro Puntos de Calibración')
   ax2.set_ylabel('Error [pix]')
   ax2.set_title('Metodo: '+str( metodos[imetVect[i]] ) +' ///  Error imagen FE')

   ax10.plot(nroCal,VVVparamErr0x[:,i],'b-o',label='Ajuste en el mapa')
   ax20.plot(nroCal,VVVparamErr0y[:,i],'r-o',label='Ajuste en el mapa')
   ax3.plot(nroCal,VVVparamErr0z[:,i],'g-o',label='Ajuste en el mapa')
   
   # ax10.plot(nroCal,VVVparamErr1x[:,i],'c-o',label='Ajuste en la imagen')
   # ax20.plot(nroCal,VVVparamErr1y[:,i],'m-o',label='Ajuste en la imagen')
   # ax3.plot(nroCal,VVVparamErr1z[:,i],'k-o',label='Ajuste en la imagen')
   
   ax10.plot(nroCal,VVVparamErr2x[:,i],'r-x',label='Ajuste 2D + imagen')
   ax20.plot(nroCal,VVVparamErr2y[:,i],'b-x',label='Ajuste 2D + imagen')
   ax3.plot(nroCal,VVVparamErr2z[:,i],'m-x',label='Ajuste 2D + imagen')
   

   ax4.plot(nroCal,VVVparamErr0a[:,i],'b-o',label='Ajuste en el mapa')
   ax5.plot(nroCal,VVVparamErr0b[:,i],'r-o',label='Ajuste en el mapa')
   ax6.plot(nroCal,VVVparamErr0g[:,i],'g-o',label='Ajuste en el mapa')
   
   # ax4.plot(nroCal,VVVparamErr1a[:,i],'c-o',label='Ajuste en la imagen')
   # ax5.plot(nroCal,VVVparamErr1b[:,i],'m-o',label='Ajuste en la imagen')
   # ax6.plot(nroCal,VVVparamErr1g[:,i],'k-o',label='Ajuste en la imagen')
   
   ax4.plot(nroCal,VVVparamErr2a[:,i],'y-x',label='Ajuste 2D + imagen')
   ax5.plot(nroCal,VVVparamErr2b[:,i],'g-x',label='Ajuste 2D + imagen')
   ax6.plot(nroCal,VVVparamErr2g[:,i],'r-x',label='Ajuste 2D + imagen')
   
   
   ax10.set_xlabel('Nro Puntos de Calibración')
   ax10.set_ylabel('Error [metros]')
   ax20.set_xlabel('Nro Puntos de Calibración')
   ax3.set_ylabel('Error [metros]')
   ax3.set_xlabel('Nro Puntos de Calibración')
   ax20.set_ylabel('Error [metros]')
   ax10.set_title('Error en la posicion X_0')
   ax20.set_title('Metodo: '+str( metodos[imetVect[i]] ) +' ///  Error en la posicion Y_0')
   ax3.set_title('Error en la posicion Z_0')

   ax4.set_xlabel('Nro Puntos de Calibración')
   ax4.set_ylabel('Error [grados]')
   ax6.set_xlabel('Nro Puntos de Calibración')
   ax6.set_ylabel('Error [grados]')
   ax5.set_xlabel('Nro Puntos de Calibración')
   ax5.set_ylabel('Error [grados]')
   ax4.set_title('Error en angulo alfa.')
   ax5.set_title('Metodo: '+str( metodos[imetVect[i]] ) +'/// Error en angulo beta.')
   ax6.set_title('Error en angulo gamma.')  

   ax1.legend()
   ax2.legend()
   ax10.legend()
   ax20.legend()
   ax3.legend()
   ax4.legend()
   ax5.legend()
   ax6.legend()

   fig.savefig('Figura_optimizacion_plano_inclinado_'+str( metodos[imetVect[i]] )+'_comparacion_costos_6-10'+'.png')
   
   fig2.savefig('Figura_optimizacion_plano_inclinado_'+str( metodos[imetVect[i]] )+'_param_Error_6-10'+'.png')
  
  






