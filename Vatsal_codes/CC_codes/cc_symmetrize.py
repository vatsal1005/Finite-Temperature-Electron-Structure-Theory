
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                            # Routine to Symmetrize the two body residue of ground state CC #
                                                # Author: Soumi Tribedi, Anish Chakraborty, Rahul Maitra #
                                                                  # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import gc
import numpy as np
import copy as cp
import MP2

##-------------------------------------------------------------##
       #Import number of occupied and virtual orbitals#
##-------------------------------------------------------------##

occ = MP2.occ
virt = MP2.virt

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

##-------------------------------------------------------------##
            #Symmetrize triples Residue i.e. R_ijkabc
        # if a=1; b=2; c=3 then according to Levi-Civita #
          # for (1,2,3), (2,3,1) and (3,1,2), sign = +1 #
          # for (3,2,1), (1,3,2) and (2,1,3), sign = -1 #
##-------------------------------------------------------------##

def symmetrize_t3(R_ijkabc):
  R_ijkabc_new = np.zeros((occ,occ,occ,virt,virt,virt))
  for i in range(0,occ):
    for j in range(0,occ):
      for k in range(0,occ):
        for a in range(0,virt):
          for b in range(0,virt):
            for c in range(0,virt):
              R_ijkabc_new[i,j,k,a,b,c] = (R_ijkabc[i,j,k,a,b,c] + R_ijkabc[j,k,i,b,c,a] + R_ijkabc[k,i,j,c,a,b] + R_ijkabc[j,i,k,b,a,c] + R_ijkabc[i,k,j,a,c,b] + R_ijkabc[k,j,i,c,b,a])#/6

  R_ijkabc = cp.deepcopy(R_ijkabc_new)

  return R_ijkabc
  R_ijkabc = None
  R_ijkabc_new = None
  gc.collect()
                          ##-----------------------------------------------------------------------------------------------------------------------##
                                                                                    #THE END#
                          ##-----------------------------------------------------------------------------------------------------------------------##
