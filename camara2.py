# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:15:51 2020

@author: hernan
"""
import numpy as np
import sim
import cv2


class Camara:
    	
    """	
    	Clase para representar una cámara arbitraria
    	Esta estructura maneja la conversión entre coordenadas de camara y coordenadas del mundo.
    """
    num_camaras = 0
    def __init__(self, tipo, xo=75., yo=75., zo=-20, alpha0= 0., beta0= 0., gamma0= 0.,k0=952.16):
		
		# Inicializo el modelo de cámara 
        
        Camara.num_camaras += 1
        self.tipo= tipo
		
        self.tampixfe=1;
		
        self.k=  k0	#	distancia focal expresada en píxeles.
        self.c=  np.array([ 960, 960])	#	Punto principal     (en px)

        self.xo= xo	# coord. X de FE en trama {m=0} medida en [m]
        self.yo= yo	# coord. Y de FE en trama {m=0} medida en [m]
        self.zo= zo	# coord. Z de FE en trama {m=0} medida en [m]
		
        self.pos_cam_w= np.asarray([self.xo, self.yo, self.zo])	#	posicion de la camara en el sistema mundo
		
        self.alfa= alpha0*np.pi/180	#	Angulos de Euler ZYX para la orientacion de la camara FE en trama {0}
        self.beta=  beta0*np.pi/180
        self.gama= gamma0*np.pi/180
		
		#	matriz de rotación (3x3) del sistema cámara al sistema mundo c->w
        self.mat_rotacion_cw= sim.eulZYX2R(self.alfa, self.beta, self.gama)

		#	matriz de rotación del sistema cámara al sistema mundo c->w
        self.mat_thomogenea_cw= np.eye(4)
        self.mat_thomogenea_cw[0:3,0:3]= self.mat_rotacion_cw
        self.mat_thomogenea_cw[0:3, 3 ]= self.pos_cam_w

		#	matriz de rotación del sistema mundo al sistema de la cámara w->c
        self.mat_thomogenea_wc= np.eye(4)
        self.mat_thomogenea_wc[0:3,0:3]= self.mat_rotacion_cw.T #matriz de rotacion de 3x3
        self.mat_thomogenea_wc[0:3, 3 ]=-self.mat_thomogenea_wc[0:3,0:3].dot(self.pos_cam_w)

		
        self.parametros  = [self.xo,   self.yo,   self.zo,   self.alfa,   self.beta,   self.gama,   self.c[0],   self.c[1],   self.k  ]
		
		
        
        
    def __del__(self):
        Camara.num_camaras -= 1	

    def calibrar_params_instrinsecos(self):
        print("calibar los parametros instrinsecos")

    def calibrar_params_extrinsecos(self):
        print("calibar los parametros extrinsecos")
		
    def camera_info(self):
		
        print("P A R A M E T R O S   I N T R I N S E C O S")
        print("tamaño del pixel: {:.3f}".format(self.tampixfe))
        print("distancia focal:  {:.3f} px".format(self.k))
        print("centro de la imagen:  ({0:.3f},{1:.3f})".format(self.c[0], self.c[1]))
        print("\n")
        print("P A R A M E T R O S   E X T R I N S E C O S")
        print("posicion de la cámara en el sistema mundo: ({0:.3f},{1:.3f},{2:.3f})".format(self.pos_cam_w[0],self.pos_cam_w[1],self.pos_cam_w[2]))
        print("matriz de rotacion mundo->camara: [{0:+.3f},{1:+.3f},{2:+.3f}".format(self.mat_thomogenea_cw[0,0],self.mat_thomogenea_cw[0,1],self.mat_thomogenea_cw[0,2]))
        print("                               {0:+.3f},{1:+.3f},{2:+.3f}".format(self.mat_thomogenea_cw[1,0],self.mat_thomogenea_cw[1,1],self.mat_thomogenea_cw[1,2]))
        print("                               {0:+.3f},{1:+.3f},{2:+.3f}]".format(self.mat_thomogenea_cw[2,0],self.mat_thomogenea_cw[2,1],self.mat_thomogenea_cw[2,2]))
		
		
    def update_rotacion(self, R):
		#	matriz de rotación del sistema cámara al sistema mundo c->w
        self.mat_rotacion_cw= R

		#	matriz de rotación del sistema cámara al sistema mundo c->w
        self.mat_thomogenea_cw= np.eye(4)
        self.mat_thomogenea_cw[0:3,0:3]= self.mat_rotacion_cw
        self.mat_thomogenea_cw[0:3, 3 ]= self.pos_cam_w

		#	matriz de rotación del sistema mundo al sistema de la cámara w->c
        self.mat_thomogenea_wc= np.eye(4)
        self.mat_thomogenea_wc[0:3,0:3]= self.mat_rotacion_cw.T #matriz de rotacion de 3x3
        self.mat_thomogenea_wc[0:3, 3 ]=-self.mat_thomogenea_wc[0:3,0:3].dot(self.pos_cam_w)
		
		#	falta acrtualizar el valor de los angulos	
        
    def update_parametros(self, P):   
        xo1=P[0]
        yo1=P[1]
        zo1=P[2]
        alpha1=P[3]
        beta1=P[4]
        gamma1=P[5]
        self.xo= xo1	# coord. X de FE en trama {m=0} medida en [m]
        self.yo= yo1	# coord. Y de FE en trama {m=0} medida en [m]
        self.zo= zo1	# coord. Z de FE en trama {m=0} medida en [m]        
        self.pos_cam_w= np.asarray([self.xo, self.yo, self.zo])
        
        self.alfa= alpha1*np.pi/180	#	Angulos de Euler ZYX para la orientacion de la camara FE en trama {0}
        self.beta=  beta1*np.pi/180
        self.gama= gamma1*np.pi/180       
        
        
        R=sim.eulZYX2R(self.alfa, self.beta, self.gama)
        
        self.update_rotacion(R)
        
        self.parametros  = [self.xo,   self.yo,   self.zo,   self.alfa,   self.beta,   self.gama,   self.c[0],   self.c[1],   self.k  ]
		
                
        
        
        
    def WorldToCamera_Proyect(self, P):
		
        Tmrcfe=self.mat_thomogenea_wc
			
        P= P.reshape( 3,-1,)
        l= np.size(P,1)
		
        Pmfe=np.ones((l,4))
        Pmfe[:,0]=P[0]
        Pmfe[:,1]=P[1]
        Pmfe[:,2]=P[2]
        Pmfe[:,3]=np.ones((1,l))  
		
        PM=    np.dot(Tmrcfe,Pmfe.T)  
        R=     np.linalg.norm(PM[0:3,:],2,axis=0)
        phi=   np.arctan2( PM[1,:], PM[0,:] )# arctan (Y/X)
        theta= np.arccos( PM[2,:] / R)# acos (Z/R)
		
        r= self.k* np.tan(theta/2)
        r1= np.multiply(r[:,np.newaxis], np.array([ np.cos(phi),np.sin(phi)]).T)  
        pix = (r1/self.tampixfe)+ self.c
		 
        return np.round(pix)
    
    
    def CameraToWorld_Proyect(self, pixel):
		
		#calculos viejo
        p=  pixel- self.c[np.newaxis,:]
        r=       np.linalg.norm(p,axis=1)

		#calculo nueva versión (solo se requiere p y r):
        rayo_c1= np.array([2*self.k*p[:,0],2*self.k*p[:,1],self.k**2-r**2])/(self.k**2+r**2)
        rayo_w= self.mat_rotacion_cw.dot(rayo_c1)
    
        return np.hstack((self.pos_cam_w[:,np.newaxis], rayo_w)) 

    def CameraToWorld_Proyect_Z0(self, vxi,vyi):
            
             T=self.mat_thomogenea_cw
        
             #calculos auxiliares sobre la imagen panoramica
             xcen=vxi-self.c[0]
             ycen=vyi-self.c[1]
    
             r = np.sqrt(xcen**2+ycen**2)
   
   
             fi = np.arctan2(ycen,xcen)
  
             #ahora a pasar a esfericas en el marco de referencia de la camara
             t = 2*np.arctan(r/self.k)
  
             Ct = np.cos(t) #coseno de theta
             St = np.sin(t) # seno de theta
  
             Cfi = np.cos(fi)
             Sfi = np.sin(fi)
  
             Px=self.xo
             Py=self.yo
             Pz=self.zo
  
             # salteandome las cartesianas paso directamente al mapa
             Rho = -(Pz) / (St*(T[2,0]*Cfi+T[2,1]*Sfi)+Ct*T[2,2])
             xM = Rho*(St*(T[0,0]*Cfi+T[0,1]*Sfi)+T[0,2]*Ct) + Px
             yM = Rho*(St*(T[1,0]*Cfi+T[1,1]*Sfi)+T[1,2]*Ct) + Py

             return xM,yM
         
            
 



    
            #     def CameraToWorld_Proyect(self, pixel):
# 		   
#         #calculos auxiliares sobre la imagen panoramica	(pixel es equivalente a vxi,vyi)
#         p=  pixel- self.c[np.newaxis,:]
# 			
# 		#ahora a pasar a esfericas en el marco de referencia de la camara
#         r=       np.linalg.norm(p,axis=1)
#         phi=     np.arctan2(p[:,1],p[:,0])
#         theta= 2*np.arctan (r/self.k)
# 		
#         cthe, cphi= np.cos([theta, phi])	#coseno de theta - phi
#         sthe, sphi= np.sin([theta, phi])	#  seno de theta - phi
# 		
# 		#rayo (en coordenadas homogeneas) en el sistema camara:
#         rayo_c= np.array([sthe*cphi, sthe*sphi, cthe]) 	
#         rayo_w= self.mat_rotacion_cw.dot(rayo_c)		
# 		
#         return np.hstack((self.pos_cam_w[:,np.newaxis], rayo_w)) 

