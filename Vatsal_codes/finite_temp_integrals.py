"""
Created on Wed Sep 28 17:07:23 2022

author: vatsal
"""

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import numpy as np
import math
from basis_transform import *
from input import *
import gc
import copy as cp
from pyscf import mp

##----------------------------------------------------------------------##
                  #Set up the denominator and t/s#
##----------------------------------------------------------------------##

D1 = np.zeros((occ,virt))
D2 = np.zeros((occ,occ,virt,virt))
t2 = np.zeros((occ,occ,virt,virt))
t1 = np.zeros((occ,virt))

for i in range(0,occ):
  for j in range(0,occ):
    for a in range(occ,nao):
      for b in range(occ,nao):
        D2[i,j,a-occ,b-occ] = hf_mo_E[i] + hf_mo_E[j] - hf_mo_E[a] - hf_mo_E[b]
        t2[i,j,a-occ,b-occ] = twoelecint_mo[i,j,a,b]/D2[i,j,a-occ,b-occ]
        
for i in range(0,occ):
  for a in range(occ,nao):
    D1[i,a-occ] = hf_mo_E[i] - hf_mo_E[a]
    t1[i,a-occ] = Fock_mo[i,a]/D1[i,a-occ]

##-------------------------------------------------------------------------------------##
                          #Calculation of MP2 energy#
##-------------------------------------------------------------------------------------##

E_mp2 = 2*np.einsum('ijab,ijab',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
print ("MP2 correlation energy is : "+str(E_mp2))
E_mp2_tot = E_hf + E_mp2
print ("MP2 energy is : "+str(E_mp2_tot))

##----------------------------------------------------------------------##
                  #redefining FT-CC integrals#
##----------------------------------------------------------------------##

#for electron and holes with opposite chemical potential, (using canonical Fock energies)

f_temp = cp.deepcopy(oneelecint_mo)
v_temp = cp.deepcopy(twoelecint_mo)

def fermi_dirac(beta,mu):
    n_FD = np.zeros(nao)
    f = np.zeros_like(f_temp)
    v = np.zeros_like(v_temp)
    
    for i in range(0,occ):      
      n_FD[i] = 1.0/(np.exp(beta*(hf_mo_E[i]-mu)) + 1.0)
      f[i,i] = n_FD[i] * (f_temp[i,i] - hf_mo_E[i])
    
    for i in range(occ,virt):
      #n_FD[i] = 1.0/(np.exp(beta*(hf_mo_E[i]-mu)) + 1.0) # same chemical potential for elctrons and holes
      n_FD[i] = 1.0/(np.exp(beta*(hf_mo_E[i]+mu)) + 1.0) # opposite chemical potential for elctrons and holes
      f[i,i] = (1 - n_FD[i]) * (f_temp[i,i] - hf_mo_E[i])
      
      
    
    for i in range(0,occ):
      for j in range(0,occ):
          if j!=i:
              f[i,j] = f_temp[i,j] * math.sqrt(n_FD[i]*n_FD[j])
          for k in range(0,occ):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*n_FD[j]*n_FD[k]*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*n_FD[j]*n_FD[k]*(1.0-n_FD[l]))
          for k in range(occ,virt):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*n_FD[j]*(1.0-n_FD[k])*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*n_FD[j]*(1.0-n_FD[k])*(1.0-n_FD[l]))
      for j in range(occ,virt):
          f[i,j] = f_temp[i,j] * math.sqrt(n_FD[i]*(1.0-n_FD[j]))
          f[j,i] = f_temp[j,i] * math.sqrt(n_FD[i]*(1.0-n_FD[j]))
          for k in range(0,occ):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*(1.0-n_FD[j])*n_FD[k]*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*(1.0-n_FD[j])*n_FD[k]*(1.0-n_FD[l]))
          for k in range(occ,virt):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*(1.0-n_FD[j])*(1.0-n_FD[k])*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt(n_FD[i]*(1.0-n_FD[j])*(1.0-n_FD[k])*(1.0-n_FD[l]))
    
    for i in range(occ,virt):
      for j in range(0,occ):
          for k in range(0,occ):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*n_FD[j]*n_FD[k]*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*n_FD[j]*n_FD[k]*(1.0-n_FD[l]))
          for k in range(occ,virt):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*n_FD[j]*(1.0-n_FD[k])*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*n_FD[j]*(1.0-n_FD[k])*(1.0-n_FD[l]))
      for j in range(occ,virt):
          f[i,j] = f_temp[i,j] * math.sqrt((1.0-n_FD[i])*(1.0-n_FD[j]))
          for k in range(0,occ):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*(1.0-n_FD[j])*n_FD[k]*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*(1.0-n_FD[j])*n_FD[k]*(1.0-n_FD[l]))
          for k in range(occ,virt):
              for l in range(0,occ):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*(1.0-n_FD[j])*(1.0-n_FD[k])*n_FD[l])
              for l in range(occ,virt):
                  v[i,j,k,l] = v_temp[i,j,k,l] * math.sqrt((1.0-n_FD[i])*(1.0-n_FD[j])*(1.0-n_FD[k])*(1.0-n_FD[l]))
      
    return n_FD, f, v

m = mp.MP2(mf)
def check_mp2():
  if abs(m.kernel()[0]-E_mp2) <= 1E-6:
    print ("MP2 successfully done")
  return

check_mp2()

gc.collect()
