# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:47:17 2020

@author: hernan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
np.set_printoptions(suppress=True)
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

FlagRuidoPARAM=1# Flag de incerteza en los parametros FE


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
  # DistanciasX_costo=np.linalg.norm(aux[0],axis=0)
  # DistanciasY_costo=np.linalg.norm(aux[1],axis=0)

  # DistanciaTotal_costo= DistanciasX_costo+DistanciasY_costo

  return DistanciaTotal_costo



#######################################################################################################################
#      Defino la superficie 'real' y un plano cuyos parametros van a ser optimizados para ajustar la superficie
#######################################################################################################################

######################################################
#Parametros iniciales del plano y la superficie "real"
######################################################
#plano a optimizar

n1=0
n2=-0.005
n3=0.75




normalPlano=np.array([n1, n2,n3])
z0Plano=-1

#superficie real 
a=0.0094
mu=0.73
h=1

######################################################

#Defino la cantidad de puntos a utilizar en la calibracion
nroCalibracion=20

print('--------------------------------------------------')
print('              OPTIMIZACION')
print('--------------------------------------------------')

print('Nro Puntos de calibracion: ' + str(nroCalibracion))
print('\n')
 
 
m=nroCalibracion
VMIN=0
VMAX=100



#Defino el nro de  iteraciones
iteraciones=20

tol = 1e-7
tol2 = 0.001

options={'maxiter' : 20000}

#estos son todos los metodos que tiene cargados el optimize.minimize
metodos=['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP','trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']

#lista de los metodos que voy a probar
imetVect=[3] 
#genero los puntos a partir del generador rs , partiendo de una semilla, la cual se guarda en el disco
semilla=np.random.seed()

rs=np.random.RandomState(semilla)


# Genero una matriz de  numeros semialeatorios que van de VMIN a VMAX

muB=50
sigmaB=20

xy=rs.normal(muB, sigmaB, (nroCalibracion,2))


X=xy[:,0]
Y=xy[:,1]

#altura de la superficie real
# Z=-sim.f2(X,Y,a,mu,h)


#genero el plano inicial
p1=sim.Plano(normalPlano,[X,Y],z0Plano)

Z=p1.z


pars= np.hstack(([p1.normal[0],p1.normal[1],p1.z0] ,paramFE_R))

#defino la camara cuyos parametros van a ir siendo optimizados
FE_Rcosto=camara2.Camara(2, xo=pars[3], yo=pars[4], zo=pars[5], alpha0=pars[6], beta0=pars[7], gamma0=pars[8])


FE_Opt=camara2.Camara(2, xo=pars[3], yo=pars[4], zo=pars[5], alpha0=pars[6], beta0=pars[7], gamma0=pars[8],k0=FE.k)
#puntos de calibracion 
r2=np.array([X,Y,Z])


#proyecto los puntos a rayos en coordenadas de la camara
rayo_pix=FE.WorldToCamera_Proyect(r2)


np.save('semilla_planoInclinado',semilla)
np.save('RandomState_planoInclinado',rs)

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



#%%#########################################################################################################
                    #OPTIMIZACION / MINIMIZACION DEL ERROR 
#########################################################################################################



for i in range(len(imetVect)): #Recorro los metodos de minimizacion 

  VXY_V=[]
  Vpars=[]
  Vcosto=[]
  VmapError=[]
  VimgError=[]
  VmapErrorVal=[]
  VimgErrorVal=[]
  vE_map3D=[]
  vE_map3Dval=[]

  for k in range(iteraciones):# Se repite el proceso para analizar la estabilidad del metodo
    
   print('Nro iteracion: '+ str(k+1)+ '/'+str(iteraciones))


# ####### Defino el punto al azar como el punto de validacion ######

   VXY=rs.normal(muB, sigmaB, (2))

   XV=VXY[0]
   YV=VXY[1]
  
   Z_V= -sim.f2(XV,YV,a,mu,h)

   #Z_V= p1.evaluate(XV,YV)

   #punto de validacion 
   r2V=np.array([XV,YV,Z_V])  


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
   F0=costo(pars, rayo_pixi ) #COSTO INICIAL

   FF=True
   OPTvector=[]


##### INICIO  del bucle  de minimizacion ######################################
   while (FF):

    ret1 = minimize(costo, pars, args=(rayo_pixi), method=metodos[imet], tol=tol)

  
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


################     VALIDACION     ###########################################

 




   rayo_pix_V=FE.WorldToCamera_Proyect(r2V)

   rayo_pix_Vf=rayo_pix_V+ruido_Pixel_Validacion
   
   #retroproyecto los rayos al mundo
   V_mF_V=FE_Opt.CameraToWorld_Proyect(rayo_pix_Vf)


   # interseco los rayos con el plano a optimizar
   r1opt_V=sim.interseccion(V_mF_V,p1opt)

 
   rayo1_pix_OPT_V=FE.WorldToCamera_Proyect(r1opt_V)  

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

   #Error 3D (validacion)
   DistanciaTotal3D_V=np.linalg.norm(r1opt_V.T-r2V,axis=1)

   #Error en mapa satelital (validacion)
   DistanciaTotalMapa_V=np.linalg.norm(r1opt_V[0:2].T-r2V[0:2],axis=1)

   #Error en imagen FE (validacion)
   ppV=rayo_pix_V
   pp1V=rayo1_pix_OPT_V
   DistanciaTotalImg_V=(np.linalg.norm(  (pp1V-ppV),axis=1))
   
   VimgErrorVal.append(DistanciaTotalImg_V)
   VmapErrorVal.append (DistanciaTotalMapa_V)
   vE_map3Dval.append(DistanciaTotal3D_V)
       
   VimgError.append( DistanciaTotalImg ) 
   VmapError.append(DistanciaTotalMapa)
   vE_map3D.append(DistanciaTotal3D )
       
#Error en los parametros de la camara FE
   E_PARAM=abs(np.mean(Vpars,axis=0)[3:9]-paramFE)
   E_PARAM_I=abs(paramFE_R_INICIAL-paramFE)
 
  print('\n')
  print('-----------------------------------')
  print(str(metodos[imet]))
  print('-----------------------------------')
  print('Promedio de error en mapa [metros] (Ptos calibracion ):   ',np.mean(VmapError))
  print('Promedio de error en imagen FE [pixeles] (Ptos calibracion ):   ',np.mean(VimgError))
  print('Promedio de error 3D [metros] (Ptos calibracion ):   ',np.mean(vE_map3D))
  print('-----------------------------------')
  # print('Promedio de error en mapa [metros] (Ptos validacion ):   ',np.mean(VmapErrorVal))
  # print('Promedio de error en imagen FE [pixeles] (Ptos validacion ):   ',np.mean(VimgErrorVal))
  # print('Promedio de error 3D [metros] (Ptos validacion ):   ',np.mean(vE_map3Dval))
  print('-----------------------------------')
  print('ParamFE optimizados:',np.mean(Vpars,axis=0)[3:9])
  print('Error en ParamFE Iniciales:',E_PARAM_I)
  print('Error en ParamFE optimizados:',E_PARAM)
  print('-----------------------------------')
  




  fig=plt.figure(figsize=(10,10)) 
  fig.subplots_adjust(hspace=0.75)   
  
  ax1 = fig.add_subplot(411)
  ax1.plot(np.arange(iteraciones)+1,VmapError,'b-o',linewidth=3,label='Error en Ptos Calibracion')
  ax1.plot(np.arange(iteraciones)+1,np.mean(VmapError)*np.ones((iteraciones,1)),'b-.',label='Promedio: '+': {:,.4f}'.format(np.mean(VmapError)))
  ax1.plot(np.arange(iteraciones)+1,VmapErrorVal,'c-o',linewidth=3,label='Error en Ptos Validacion')
  ax1.plot(np.arange(iteraciones)+1,np.mean(VmapErrorVal)*np.ones((iteraciones,1)),'c-.',label='Promedio: '+': {:,.4f}'.format(np.mean(VmapErrorVal))  )
  ax1.set_ylabel('Error [metros]')
  ax1.set_title('Modelo del plano inclinado //Error en el mapa //  Método de optimizacion:'+ str(metodos[imet]))
  ax1.legend()
  ax1.set_xlim([1,iteraciones])

  ax2 = fig.add_subplot(412)
  ax2.plot(np.arange(iteraciones)+1,VimgError,'r-o',linewidth=3,label='Error en Ptos Calibracion')
  ax2.plot(np.arange(iteraciones)+1,np.mean(VimgError)*np.ones((iteraciones,1)),'r-.',label='Promedio: '+': {:,.4f}'.format(np.mean(VimgError)))
  ax2.plot(np.arange(iteraciones)+1,VimgErrorVal,'m-o',linewidth=3,label='Error en Ptos Validacion')
  ax2.plot(np.arange(iteraciones)+1,np.mean(VimgErrorVal)*np.ones((iteraciones,1)),'m-.',label='Promedio: '+': {:,.4f}'.format( np.mean(VimgErrorVal)))
  ax2.set_ylabel('Error [pixeles]')
  ax2.set_title('Error en la imagen FE')
  ax2.legend()
  ax2.set_xlim([1,iteraciones])

  ax3 = fig.add_subplot(413)
  ax3.plot(np.arange(iteraciones)+1,vE_map3D,'g-o',linewidth=3,label='Error en Ptos Calibracion')
  ax3.plot(np.arange(iteraciones)+1,np.mean(vE_map3D)*np.ones((iteraciones,1)),'g-.',label='Promedio: '+': {:,.4f}'.format(np.mean(vE_map3D)))
  ax3.plot(np.arange(iteraciones)+1,vE_map3Dval,'k-o',linewidth=3,label='Error en Ptos Validacion')
  ax3.plot(np.arange(iteraciones)+1,np.mean(vE_map3Dval)*np.ones((iteraciones,1)),'k-.',label='Promedio: '+': {:,.4f}'.format( np.mean(vE_map3Dval)))
  ax3.set_xlabel('nro iteraciones'+'Error en Param FE optimizados:'+str(E_PARAM))
  ax3.set_ylabel('Error [metros]')
  ax3.set_title('Error 3D')
  ax3.legend()
  ax3.set_xlim([1,iteraciones])