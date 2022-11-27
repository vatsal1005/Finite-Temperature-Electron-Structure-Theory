"""
Created on Thu Sep 29 03:56:05 2022

author: vatsal
"""

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import numpy as np
import copy as cp
from finite_temp_integrals import *
from basis_transform import *
from input import *
from FT_diagrams import *
from scipy import integrate
from matplotlib.pyplot import *
import sys
sys.path.insert(0, '/home/theomolsci/Desktop/Vatsal/FT-CC_codes/Vatsal_codes/CC_codes')
from main import E_ccd # correlation energy for CC
        
##----------------------------------------------------------------------------------------##
                   #calculation of FT-CC energies#
##----------------------------------------------------------------------------------------##

def energy_FT_ccd(s2,v):
  E_ft_cc = 2*np.einsum('ijab,ijab',s2,v[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',s2,v[:occ,:occ,occ:nao,occ:nao])
  return E_ft_cc

def energy_FT_ccsd(s1,s2,v):
  E_ft_cc = energy_FT_ccd(s2,v)
  E_ft_cc += 2*np.einsum('ijab,ia,jb',v[:occ,:occ,occ:nao,occ:nao],s1,s1) - np.einsum('ijab,ib,ja',v[:occ,:occ,occ:nao,occ:nao],s1,s1)
  return E_ft_cc

##--------------------------------------------------##
          #Defining useful parameters#
##--------------------------------------------------##

delta = 1.0/(n_steps - 1)

temp_span = temp_max - temp_min
temp_arr = [temp_min + (i * delta) ** spacing_parameter * temp_span for i in range(n_steps)]

n_g_span = n_g_max - n_g_min
n_g_arr = [int(n_g_max - (i * delta) ** spacing_parameter * n_g_span) for i in range(n_steps)]

##---------------------------------------------------------------------------##
                    #Calculation for FT-CC method#
##---------------------------------------------------------------------------##

E_FTCC_final = np.zeros(len(temp_arr)) # difference between HF and FT-CC energies
grand_FTCC_final = np.zeros(len(temp_arr))

for y in range(len(temp_arr)):
    
    temperature = temp_arr[y]
    n_g = n_g_arr[y]
    grid = np.flip([(k_b * temperature - (i * (1.0/(n_g-1))) ** spacing_parameter * (k_b * temperature)) for i in range(n_g)])
    mu = k_b * temperature * (nelec/var_N - 2)
    
    E_old = E_mp2_tot
    
    # s1 = np.zeros((occ,virt))
    # s2 = np.zeros((occ,occ,virt,virt))
    s1 = cp.deepcopy(t1)
    s2 = cp.deepcopy(t2)
    
    E_FTCC = np.zeros(n_g)
    
    for x in range(len(grid)):
      beta = grid[x]
      n_FD, f, v = fermi_dirac(beta,mu)
      
      tau = cp.deepcopy(s2)
      I_vv, I_oo, I_vo, I_ov, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = initialize_FT(f,v)
      I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = update_int_FT(tau,s2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov,v)
      
      if calc == 'FTCCD':
        print ("\n-----------FT-CCD------------")
        print (str(temperature)+" K (Step "+str(y+1)+"/"+str(n_steps)+")")
        
        S_ijab = doubles_FT(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,s2,s1,v)
        S_ijab = symmetrize(S_ijab)
        
        eps_s, s2 = update_FT_s2(S_ijab,s2,x,grid)
        E_ft_cc = energy_FT_ccd(s2,v)
        
      if calc == 'FTCCSD':
        print ("\n-----------FT-CCSD------------")
        print (str(temperature)+" K (Step "+str(y+1)+"/"+str(n_steps)+")")
        
        tau += np.einsum('ia,jb->ijab',s1,s1)
        I1, I2 = R_ia_intermediates_FT(s1,v)
        S_ia = singles_FT(I1, I2, I_oo, I_vv, tau, s1, s2, f, v)
        I_oo, I_vv, I_oovo, I_vovv, Ioooo_2, I_voov, Iovov_3, Iovvo_3, Iooov, I3 = singles_intermediates_FT(s1, s2, I_oo, I_vv, I2, v)
        
        S_ijab = doubles_FT(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,s2,s1,v)
        S_ijab += singles_n_doubles_FT(t1, I_oovo, I_vovv, v)
        S_ijab += higher_order(t1, t2, Iovov_3, Iovvo_3, Iooov, I3, Ioooo_2, I_voov)
        S_ijab = symmetrize(S_ijab)
        
        eps_s, s1, s2 = update_FT_s1s2(S_ia, S_ijab, s1, s2, x, grid)
        E_ft_cc = energy_FT_ccsd(s1, s2, v)
        
      del_E = E_ft_cc - E_old
      E_old = E_ft_cc
      E_FTCC[x] = E_hf + E_ft_cc
      print ("cycle number: "+str(x+1))
      print ("||S_ijab||: "+str(eps_s))
      print ("energy difference: "+str(del_E))
      print ("E_HF + E_"+str(calc)+": "+str(E_FTCC[x]))
      
    def f_grand(beta): # Quadrature for calculating grand potential
      x = (np.abs(np.asarray(grid)-beta)).argmin()
      if E_FTCC[x] == E_FTCC[0]:
          f_quad = k_b * temperature * (1.0/2) * (E_FTCC[x] + E_FTCC[x+1])
      elif E_FTCC[x] == E_FTCC[-1]:
          f_quad = k_b * temperature * (1.0/2) * (E_FTCC[x-1] + E_FTCC[x])
      else:
          f_quad = k_b * temperature * (1.0/3) * (E_FTCC[x-1] + 4 * E_FTCC[x] + E_FTCC[x+1])
      return f_quad

    grand, grand_err = integrate.quadrature(np.vectorize(f_grand), 0.0, 1.0/(k_b * temperature))
    print ("\ngrand potential: "+str(grand))
    print ("error in grand potential: "+str(grand_err))
    
    E_FTCC_final[y] = E_FTCC[-1]
    grand_FTCC_final[y] = grand

##---------------------------------------------------------------------------##
                    #Plotting finite temperature results#
##---------------------------------------------------------------------------##

figure(1)
plot(temp_arr, 27.211386245988 * (E_FTCC_final - E_hf - E_ccd), color='C2', label =str(calc)+' energy - '+str(calc[2:])+' energy)', marker='+')
ylabel('Energy [eV]')
xlabel('Temperature [K]')
title('Energy ('+str(calc)+') vs temperature for '+str(mol.atom.partition(' ')[0])+' ('+str(temp_min)+' K to '+str(temp_max)+' K)')
legend()
savefig('results/'+str(mol.atom.partition(' ')[0])+'_'+str(temp_min)+'K_to_'+str(temp_max)+'K_energy_'+str(calc)+'.pdf',bbox_inches='tight',pad_inches=0.2)

figure(2)
plot(temp_arr, 27.211386245988 * (E_FTCC_final - E_hf), color='C0', label =str(calc)+' energy - HF energy', marker='x')
ylabel('Energy [eV]')
xlabel('Temperature [K]')
title('Correlation energy ('+str(calc)+') vs temperature for '+str(mol.atom.partition(' ')[0])+' ('+str(temp_min)+' K to '+str(temp_max)+' K)')
legend()
savefig('results/'+str(mol.atom.partition(' ')[0])+'_'+str(temp_min)+'K_to_'+str(temp_max)+'K_correlation_energy_'+str(calc)+'.pdf',bbox_inches='tight',pad_inches=0.2)

figure(3)
plot(temp_arr, grand_FTCC_final, color='C4', label ='Grand potential ('+str(calc)+')', marker='o')
ylabel('Grand potential ('+str(calc)+')')
xlabel('Temperature [K]')
title('Grand potential ('+str(calc)+') vs temperature for '+str(mol.atom.partition(' ')[0])+' ('+str(temp_min)+' K to '+str(temp_max)+' K)')
savefig('results/'+str(mol.atom.partition(' ')[0])+'_'+str(temp_min)+'K_to_'+str(temp_max)+'K_grand_potential_'+str(calc)+'.pdf',bbox_inches='tight',pad_inches=0.2)
