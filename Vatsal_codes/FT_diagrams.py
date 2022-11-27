"""
Created on Thu Sep 29 11:37:04 2022

author: vatsal
"""

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import gc
import numpy as np
import copy as cp
from finite_temp_integrals import *
from basis_transform import *
from input import *

##----------------------------------------------------------------------------------------------##
      #Initialize intermediates for linear terms in FT-CCD. LCCD is the default calculation#
##----------------------------------------------------------------------------------------------##

def initialize_FT(f,v):
  I_vv = cp.deepcopy(f[occ:nao,occ:nao])
  I_oo = cp.deepcopy(f[:occ,:occ])
  I_vo = cp.deepcopy(f[occ:nao,:occ])
  I_ov = cp.deepcopy(f[:occ,occ:nao])
  Ivvvv = cp.deepcopy(v[occ:nao,occ:nao,occ:nao,occ:nao])
  Ioooo = cp.deepcopy(v[:occ,:occ,:occ,:occ])
  Iovvo = cp.deepcopy(v[:occ,occ:nao,occ:nao,:occ])
  Iovvo_2 = cp.deepcopy(v[:occ,occ:nao,occ:nao,:occ])
  Iovov = cp.deepcopy(v[:occ,occ:nao,:occ,occ:nao])
  Iovov_2 = cp.deepcopy(v[:occ,occ:nao,:occ,occ:nao])
  return I_vv, I_oo, I_vo, I_ov, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov, Iovov_2

  I_vv = None
  I_oo = None
  I_vo = None
  I_ov = None
  Ivvvv = None
  Ioooo = None
  Iovvo = None
  Iovvo_2 = None
  Iovov = None
  Iovov_2 = None
  I_oovo = None
  I_vovv = None
  gc.collect()
  
##----------------------------------------------------------------------------------------------##
                       #[FT] Introducing the non-linear doubles terms#
##----------------------------------------------------------------------------------------------##

def update_int_FT(tau,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov,v):
  I_vv += -2*np.einsum('cdkl,klad->ca',v[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdkl,klda->ca',v[occ:nao,occ:nao,:occ,:occ],t2)

  I_oo += 2*np.einsum('cdkl,ilcd->ik',v[occ:nao,occ:nao,:occ,:occ],tau) - np.einsum('dckl,lidc->ik',v[occ:nao,occ:nao,:occ,:occ],tau) 

  Ioooo += np.einsum('cdkl,ijcd->ijkl',v[occ:nao,occ:nao,:occ,:occ],t2)

  Iovvo += np.einsum('dclk,jlbd->jcbk',v[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',v[occ:nao,occ:nao,:occ,:occ],t2) - 0.5*np.einsum('cdlk,jlbd->jcbk',v[occ:nao,occ:nao,:occ,:occ],t2)

  Iovvo_2 += -0.5*np.einsum('dclk,jldb->jcbk',v[occ:nao,occ:nao,:occ,:occ],t2)  - np.einsum('dckl,ljdb->jcbk',v[occ:nao,occ:nao,:occ,:occ],t2)

  Iovov += -0.5*np.einsum('dckl,ildb->ickb',v[occ:nao,occ:nao,:occ,:occ],t2)
  return I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov

  I_vv = None
  I_oo = None
  Iovvo = None
  Iovvo_2 = None
  Iovov = None
  gc.collect()

##--------------------------------------------------------------------------##
      #[FT] intermediates for contribution of singles to R_ijab #
##--------------------------------------------------------------------------##

def R_ia_intermediates_FT(s1,v): 
  I1 = 2*np.einsum('cbkj,kc->bj',v[occ:nao,occ:nao,:occ,:occ],s1)
  I2 = -np.einsum('cbjk,kc->bj',v[occ:nao,occ:nao,:occ,:occ],s1)
  return I1,I2

  I1 = None
  I2 = None

def singles_intermediates_FT(t1,t2,I_oo,I_vv,I2,v):
  I_oo += 2*np.einsum('ibkj,jb->ik',v[:occ,occ:nao,:occ,:occ],t1)    #intermediate for diagrams 5
  I_oo += -np.einsum('ibjk,jb->ik',v[:occ,occ:nao,:occ,:occ],t1)     #intermediate for diagrams 8
  I_vv += 2*np.einsum('bcja,jb->ca',v[occ:nao,occ:nao,:occ,occ:nao],t1)    #intermediate for diagrams 6
  I_vv += -np.einsum('cbja,jb->ca',v[occ:nao,occ:nao,:occ,occ:nao],t1)    #intermediate for diagrams 7
  I_vv += -2*np.einsum('dclk,ld,ka->ca',v[occ:nao,occ:nao,:occ,:occ],t1,t1)  #intermediate for diagram 34'

  I_oovo = np.zeros((occ,occ,virt,occ))
  I_oovo += -np.einsum('cikl,jlca->ijak',v[occ:nao,:occ,:occ,:occ],t2)    #intermediate for diagrams 11
  I_oovo += np.einsum('cdka,jicd->ijak',v[occ:nao,occ:nao,:occ,occ:nao],t2)    #intermediate for diagrams 12
  I_oovo += -np.einsum('jclk,lica->ijak',v[:occ,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 13
  I_oovo += 2*np.einsum('jckl,ilac->ijak',v[:occ,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 15
  I_oovo += -np.einsum('jckl,ilca->ijak',v[:occ,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 17

  I_vovv = np.zeros((virt,occ,virt,virt))
  I_vovv += np.einsum('cjkl,klab->cjab',v[occ:nao,:occ,:occ,:occ],t2)    #intermediate for diagrams 9
  I_vovv += -np.einsum('cdlb,ljad->cjab',v[occ:nao,occ:nao,:occ,occ:nao],t2)    #intermediate for diagrams 10
  I_vovv += -np.einsum('cdka,kjdb->cjab',v[occ:nao,occ:nao,:occ,occ:nao],t2)    #intermediate for diagrams 14
  I_vovv += 2*np.einsum('cdal,ljdb->cjab',v[occ:nao,occ:nao,occ:nao,:occ],t2)    #intermediate for diagrams 16
  I_vovv += -np.einsum('cdal,jldb->cjab',v[occ:nao,occ:nao,occ:nao,:occ],t2)    #intermediate for diagrams 18

  Ioooo_2 = 0.5*np.einsum('cdkl,ic,jd->ijkl',v[occ:nao,occ:nao,:occ,:occ],t1,t1)    #intermediate for diagrams 37
  I_voov = -np.einsum('cdkl,kjdb->cjlb',v[occ:nao,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 39

  Iovov_3 = -np.einsum('dckl,ildb->ickb',v[occ:nao,occ:nao,:occ,:occ],t2)  #intermediate for diagrams 36

  Iovvo_3 = 2*np.einsum('dclk,jlbd->jcbk',v[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',v[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdak,ic->idak',v[occ:nao,occ:nao,occ:nao,:occ],t1)  #intermediate for diagrams 32,33,31

  Iooov = np.einsum('dl,ijdb->ijlb',I2,t2) #intermediate for diagram 34

  Iovvo_3 += -np.einsum('iclk,la->icak',v[:occ,occ:nao,:occ,:occ],t1)  #intermediate for diagram 30

  I3 = -np.einsum('cdkl,ic,ka->idal',v[occ:nao,occ:nao,:occ,:occ],t1,t1)  #intermediate for diagram 40
  return I_oo, I_vv, I_oovo, I_vovv, Ioooo_2, I_voov, Iovov_3, Iovvo_3, Iooov, I3

  I_vv = None
  I_oo = None
  I_oovo = None
  I_vovv = None
  Ioooo_2 = None
  I_voov = None
  Iovov_3 = None
  Iovvo_3 = None
  Iooov = None
  I3 = None
  gc.collect()

##--------------------------------------------------------------------##
                  #[FT] t1 and t2 contributing to R_ia#
##--------------------------------------------------------------------##

def singles_FT(I1,I2,I_oo,I_vv,tau,t1,t2,f,v):
  R_ia = cp.deepcopy(f[:occ,occ:nao])
  R_ia += -np.einsum('ik,ka->ia',I_oo,t1)                                          #diagrams 1,l,j,m,n
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)                                           #diagrams 2,k,i
  R_ia += -2*np.einsum('ibkj,kjab->ia',v[:occ,occ:nao,:occ,:occ],tau)     #diagrams 5 and a
  R_ia += np.einsum('ibkj,jkab->ia',v[:occ,occ:nao,:occ,:occ],tau)     #diagrams 6 and b
  R_ia += 2*np.einsum('cdak,ikcd->ia',v[occ:nao,occ:nao,occ:nao,:occ],tau) #diagrams 7 and c
  R_ia += -np.einsum('cdak,ikdc->ia',v[occ:nao,occ:nao,occ:nao,:occ],tau) #diagrams 8 and d
  R_ia += 2*np.einsum('bj,ijab->ia',I1,t2) - np.einsum('bj,ijba->ia',I1,t2)     #diagrams e,f
  R_ia += 2*np.einsum('bj,ijab->ia',I2,t2) - np.einsum('bj,ijba->ia',I2,t2)     #diagrams g,h
  R_ia += 2*np.einsum('icak,kc->ia',v[:occ,occ:nao,occ:nao,:occ],t1)           #diagram 3
  R_ia += -np.einsum('icka,kc->ia',v[:occ,occ:nao,:occ,occ:nao],t1)           #diagram 4
  return R_ia

  R_ia = None
  I_oo = None
  I_vv = None
  I1 = None
  I2 = None
  gc.collect()
  
##--------------------------------------------------------------------------##
                  #[FT] t2 and tau contributing to R_ijab#
##--------------------------------------------------------------------------##

def doubles_FT(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2,t1,v):
  R_ijab = 0.5*cp.deepcopy(v[:occ,:occ,occ:nao,occ:nao])
  R_ijab += -np.einsum('ik,kjab->ijab',I_oo,t2)        #diagrams linear 1 and non-linear 25,27,5,8,35,38'
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)         #diagrams linear 2 and non-linear 24,26,34',6,7
  R_ijab += -np.einsum('ijkb,ka->ijab',v[:occ,:occ,:occ,occ:nao],t1)            #diagram 3
  R_ijab += np.einsum('cjab,ic->ijab',v[occ:nao,:occ,occ:nao,occ:nao],t1)       #diagram 4
  R_ijab += 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,tau) #diagrams linear 5 and non-linear 2
  R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)    #diagrams linear 6 and non-linear 19,28,20
  R_ijab += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)   #diagram linear 7
  R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)  #diagrams linear 8 and non-linear 21,29 
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,tau)  #diagrams linear 9 and non-linear 1,22,38
  R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,t2)    #diagrams linear 10 and non-linear 23
  return R_ijab

  R_ijab = None
  I_oo = None
  I_vv = None
  Ivvvv = None
  Ioooo = None
  Iovvo = None
  Iovvo_2 = None
  Iovov = None
  Iovov_2 = None
  gc.collect()
  
##-----------------------------------------------------------------##
                #[FT] t1 terms contributing to R_ijab#
##-----------------------------------------------------------------##

def singles_n_doubles_FT(t1,I_oovo,I_vovv,v):
  R_ijab = -np.einsum('ijak,kb->ijab',I_oovo,t1)       #diagrams 11,12,13,15,17
  R_ijab += np.einsum('cjab,ic->ijab',I_vovv,t1)       #diagrams 9,10,14,16,18
  R_ijab += -np.einsum('ickb,ka,jc->ijab',v[:occ,occ:nao,:occ,occ:nao],t1,t1)   #diagrams non-linear 3
  R_ijab += -np.einsum('icak,jc,kb->ijab',v[:occ,occ:nao,occ:nao,:occ],t1,t1)   #diagrams non-linear 4
  return R_ijab

  R_ijab = None
  I_oovo = None
  I_vovv = None
  gc.collect() 

##------------------------------------------------------------------##
           #Higher orders of t1 contributing to R_ijab#
##------------------------------------------------------------------##

def higher_order(t1,t2,Iovov_3,Iovvo_3,Iooov,I3,Ioooo_2,I_voov):
  R_ijab = -np.einsum('ickb,jc,ka->ijab',Iovov_3,t1,t1)       #diagrams 36
  R_ijab += -np.einsum('jcbk,ic,ka->ijab',Iovvo_3,t1,t1)      #diagrams 32,33,31,30
  R_ijab += -np.einsum('ijlb,la->ijab',Iooov,t1)      #diagram 34,30
  R_ijab += -0.5*np.einsum('idal,jd,lb->ijab',I3,t1,t1)      #diagram 40
  R_ijab += np.einsum('ijkl,klab->ijab',Ioooo_2,t2)      #diagram 37
  R_ijab += -np.einsum('cjlb,ic,la->ijab',I_voov,t1,t1)      #diagram 39
  return R_ijab

  R_ijab = None 
  Iovov_3 = None 
  Iovvo_3 = None 
  Iooov = None 
  I3 = None 
  Ioooo_2 = None 
  I_voov = None
  gc.collect()
  
##-------------------------------------------------------------##
            #Symmetrize Residue i.e. R_ijab
##-------------------------------------------------------------##

def symmetrize(R_ijab):
  R_ijab_new = np.zeros((occ,occ,virt,virt))
  for i in range(0,occ):
        for j in range(0,occ):
                for a in range(0,virt):
                        for b in range(0,virt):
                                R_ijab_new[i,j,a,b] = R_ijab[i,j,a,b] + R_ijab[j,i,b,a]

  R_ijab = cp.deepcopy(R_ijab_new)

  return R_ijab
  R_ijab = None
  R_ijab_new = None
  gc.collect()
  
##--------------------------------------------------##
              #compute new FT s2#
##--------------------------------------------------##

def update_FT_s2(S_ijab,s2,x,grid):
  beta = grid[x]
  beta_prev = grid[x-1]
  if x==0:
      beta_prev = 0
  del_s2 = -(beta - beta_prev)*(np.add(np.multiply(D2,s2),S_ijab))
  s2 = s2 + del_s2
  eps_s = np.sum(abs(S_ijab)**4)**(1.0/4)
  return eps_s, s2

##--------------------------------------------------##
              #compute new FT s1 and s2#
##--------------------------------------------------##

def update_FT_s1s2(S_ia,S_ijab,s1,s2,x,grid):
  beta = grid[x]
  beta_prev = grid[x-1]
  if x==0:
      beta_prev = 0
  del_s2 = -(beta - beta_prev)*(np.add(np.multiply(D2,s2),S_ijab))
  del_s1 = -(beta - beta_prev)*(np.add(np.multiply(D1,s1),S_ia))
  s2 = s2 + del_s2
  s1 = s1 + del_s1
  eps_s = np.sum(abs(S_ijab)**4)**(1.0/4) + np.sum(abs(S_ia)**2)**(1.0/2)
  return eps_s, s1, s2