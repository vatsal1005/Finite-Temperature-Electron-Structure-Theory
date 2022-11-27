# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:38:52 2022

@author: asus
"""

def my_lin(lb, ub, steps, spacing=2.5): #temperatures
    span = (ub-lb)
    dx = 1.0 / (steps)
    return [lb + (i*dx)**spacing*span for i in range(steps+1)]

lin = my_lin(100, 1600, 10)
print (lin)


def my_lin_rev(lb, ub, steps, spacing=2.5): #number of grid points
    span = (ub-lb)
    dx = 1.0 / (steps)
    return [int(ub - (i*dx)**spacing*span) for i in range(steps+1)]


lin = my_lin_rev(100, 1000, 10)
print (lin)