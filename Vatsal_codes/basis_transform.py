"""
Created on Wed Sep 28 17:08:45 2022

author: vatsal
"""

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import gc
import numpy as np
import input
from pyscf import gto, scf, cc
from pyscf import ao2mo
from pyscf.ao2mo.addons import load
from pyscf import symm
from math import ceil

mol = input.mol

##--------------------------------------------------------------##
          #Import different parameters from pyscf#
##--------------------------------------------------------------##

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

nao = mol.nao_nr() # Obtain the number of atomic orbitals in the basis set

nelec = mol.nelectron # Obtain the number of electrons

enuc = mol.energy_nuc() # Compute nuclear repulsion energy

T = mol.intor('cint1e_kin_sph') # Compute one-electron kinetic integrals

V = mol.intor('cint1e_nuc_sph') # Compute one-electron potential integrals

v2e = mol.intor('cint2e_sph').reshape((nao,)*4) # Compute two-electron repulsion integrals (Chemists' notation)

##--------------------------------------------------------------##
                ####  Hartree-Fock pyscf  ####
##--------------------------------------------------------------##

if nelec % 2 == 0:
    mf = scf.RHF(mol).run()
else:
    mf = scf.ROHF(mol).run()
E_hf = mf.e_tot
hf_mo_E = mf.mo_energy
hf_mo_coeff = mf.mo_coeff

print (np.shape(hf_mo_E))

##--------------------------------------------------------------##
             #Set up initial Fock matrix#
##--------------------------------------------------------------##

Fock = T + V
n = ceil(nelec/2)
occ = n
virt = nao - n

##--------------------------------------------------------------##
      #Transform the 1 electron integrals to MO basis#
##--------------------------------------------------------------##

oneelecint_mo = np.einsum('ab,ac,cd->bd',hf_mo_coeff,Fock,hf_mo_coeff)

##--------------------------------------------------------------##
         #Transform 2 electron integrals to MO Basis#
##--------------------------------------------------------------##

if input.integral_trans == 'incore':
  twoelecint_1 = np.einsum('zs,wxyz->wxys',hf_mo_coeff,v2e)
  twoelecint_2 = np.einsum('yr,wxys->wxrs',hf_mo_coeff,twoelecint_1)
  twoelecint_3 = np.einsum('xq,wxrs->wqrs',hf_mo_coeff,twoelecint_2)
  twoelecint_mo = np.einsum('wp,wqrs->pqrs',hf_mo_coeff,twoelecint_3)
  twoelecint_1 = None
  twoelecint_2 = None
  twoelecint_3 = None

if input.integral_trans == 'outcore':
  ao2mo.outcore.full(mol,hf_mo_coeff,'2eint.h5',aosym='s4',max_memory=8000,verbose=2)

  with load('2eint.h5') as twoelecint_mo:
    twoelecint_mo = ao2mo.restore(1,twoelecint_mo,nao)
    
##--------------------------------------------------------------##
                  #Verify integrals#
##--------------------------------------------------------------##

E_scf_mo_1 = 0
E_scf_mo_2 = 0
E_scf_mo_3 = 0

for i in range(0,n):
  E_scf_mo_1 += oneelecint_mo[i][i]
for i in range(0,n):
  for j in range(0,n):
    E_scf_mo_2 += 2*twoelecint_mo[i][i][j][j] - twoelecint_mo[i][j][i][j]

Escf_mo = 2*E_scf_mo_1 + E_scf_mo_2 + enuc

##--------------------------------------------------------------##
              #Verify with pyscf HF routine#
##--------------------------------------------------------------##

def check_mo():
  if abs(Escf_mo - E_hf)<= 1E-6 :
    print ("MO conversion successful")
  return

print (Escf_mo,E_hf)
print (hf_mo_E)
print (nelec)
check_mo()

##--------------------------------------------------------------##
                    #Create Fock matrix#
##--------------------------------------------------------------##

Fock_mo = np.zeros((nao,nao))

for i in range(0,nao):
  Fock_mo[i,i] = hf_mo_E[i]

gc.collect()

##--------------------------------------------------------------##
     #Switching from chemists' to physicists' notation#
##--------------------------------------------------------------##

twoelecint_mo = np.swapaxes(twoelecint_mo,1,2)  #physicists' notation

gc.collect()