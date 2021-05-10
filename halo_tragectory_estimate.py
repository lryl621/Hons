#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:51:12 2021

@author: luke
"""
import numpy as np

def positive_roots(nums):
    """Finds positive roots of real components for calculating position of L """
    real_part = nums.real
    ret = []
    for i in range(len(real_part)):
        if real_part[i] > 0:
            ret.append(nums[i])
    return np.array(ret)

def realest_number(nums):
    """finding the real solution assumeing that one exsists"""
    imag_part = np.abs(nums.imag)
    i = np.argmin(imag_part)
    return nums[i].real

def halo_tragectory_estimate(n,L,a_1,mu,A_z):
    
    """
    Code based off Analytic Construction of Periodic Orbits about the Collinear Points, David L. Richardson 1979
    
    Equations are numbered same as they are in the paper
    Calculates first appoximation for halo orbits
    
    n = phase of orbit type 1 = 1 and type 2 =3
    L = lagrange point the orbit is about
    a_1 = distance between the larger masses
    mu = mass parameter
    A_z = z amplitude of orbit
    """
    def c_val(n,L,gamma_L,mu):
        #Calculation of the constants of the appoximation specific to L point 
        c = 0
        if (L == 1):
            c = 1/(gamma_L ** 3) * (1**n * mu + (-1)**n * (1-mu)*gamma_L ** (n+1) /((1-gamma_L)**(n+1))  )
            
        elif (L == 2):
            c = 1/(gamma_L ** 3) * ((-1)**n * mu + (-1)**n * (1-mu)*gamma_L ** (n+1) /((1+gamma_L)**(n+1))  )
        elif (L == 3):
            c = 1/(gamma_L ** 3) * (1-mu+(mu*gamma_L ** (n+1))/(1+gamma_L)**(n+1))        
        else:
            raise ValueError("L must be 1,2 or 3")
        return c



    
    #Dermines the sign of the different phase orbits in the z axis, z axis is inverted for n=3 orbits
    delta_n = 2 - n    
    #Values for a full 2pi orbit W    
    #Defines the ratio between the distance between the L point and closest mass and the distance between the two masses. 
    r_1 = 0
    if L not in [1,2]:
        raise ValueError("L must be 1,2 or 3")
    elif L == 1:
        r_1 = np.roots([1,-(3-mu),(3 - 2*mu),-mu,2*mu,-mu])
        r_1 = positive_roots(np.array(r_1))
        r_1 = realest_number(np.array(r_1))
    
    elif L == 2:
        r_1 = np.roots([1,(3-mu),(3 - 2*mu),-mu,-2*mu,-mu])
        #print(r_1)
        r_1 = positive_roots(r_1)
        r_1 = realest_number(np.array(r_1))
        #print(r_1)
    gamma_L = r_1
    #conversion of units of A_z so that the distance between the L point and closest mass is 1
    #A_x = A_x / (gamma_L *a_1)
    
    # EQ 8
    #Constants of the Legendre approximation.
    c_2 = c_val(2,L,gamma_L,mu)
    c_3 = c_val(3,L,gamma_L,mu)
    c_4 = c_val(4,L,gamma_L,mu)
    
    #linearized freqency feound from roots of polynomial associated with the DE. Only one real solution that needs to be found.
    lam = realest_number(positive_roots(np.roots([1,0,(c_2 -2),0,-(c_2-1)*(1+2*c_2)])))
    
    #found in Appendix
    
    #constants to simplify equations.
    k = 1/ (2* lam) *  (lam**2 + 1+ 2*c_2)
    d_1 = 3 * lam**2 / k  * (k*(6*lam**2 - 1) - 2*lam)
    d_2 = (8*lam **2)/k * (k*(11*lam**2 -1) - 2*lam)
    
    
    #Coeffients for terms in the x axis
    a_21 = 3 * c_3 * (k**2 -2) / (4*(1+2*c_2))
    a_22 = 3*c_3 / (4*(1+2*c_2))
    a_23 = - 3*c_3 * lam / (4*k*d_1) * (3*k**3 * lam - 6*k*(k -lam) + 4) 
    a_24 = - 3*c_3 * lam / (4*k*d_1) * (2+ 3*k*lam)
    
    
    #Coeffients for terms in the y axis
    b_21 = - 3* c_3 * lam / (2*d_1) * (3*k*lam -4)
    b_22 = 3* c_3 * lam / d_1
    
    #Coeffients for terms in the z axis
    d_21 = - c_3 / (2*lam ** 2)
    
    
    #Coeffients for terms in the x axis
    a_31 = - 9 * lam /(4*d_2) * (4*c_3* (k*a_23- b_21) + k*c_4 * (4+ k**2)) + ((9*lam**2+1-c_2)/(2*d_2)) * (3*c_3 * (2*a_23- k*b_21) + c_4*(2+3*k**2))
    a_32 = - 1/d_2 * (9*lam /4 * (4*c_3* (k*a_24- b_22) + k*c_4)+ 3/2 * (9*lam **2 +1 -c_2)*(c_3*(k*b_22+ d_21- 2*a_24) - c_4))
    
    #Coeffients for terms in the y axis
    b_31 = 3/(8*d_2) * (8*lam*(3*c_3*(k*b_21- 2*a_23) - c_4 *(2+ 3*k**2)) + (9*lam**2 + 1 +2*c_2)*(4*c_3*(k*a_23-b_21) + k*c_4*(4+k**2)))
    b_32 = 1/d_2 * (9*lam *(c_3*(k*b_22+d_21 - 2*a_24)- c_4) + 3/8*(9*lam**2 +1 + 2 * c_2) * (4*c_3*(k*a_24 - b_22) +k*c_4))
    
    #Coeffients for terms in the z axis
    d_31 = 3/(64*lam**2) * (4*c_3*a_24 + c_4)
    d_32 = 3/(64*lam**2) * (4*c_3*(a_23 - d_21) + c_4*(4 + k**2))
    
    
    #sign correction for phase of loop 1 for n=1, -1 for n =3 
    delta = lam **2 - c_2
    
    #constants to simplify equations.
    a_2 = 3/2 * c_3 * (a_24 - 2 * a_22) + 9/8 * c_4
    a_1 = -3/2 * c_3 * (2*a_21 + a_23 + 5 * d_21) - 3/8 * c_4*(12-k**2)
    
    s_1 = (1/(2*lam*(lam*(1+k**2) - 2*k))) * ((3/2)*c_3*(2*a_21*(k**2 - 2) -a_23*(k**2 +2) - 2*k*b_21) - (3/8) * c_4 * (3*k ** 4 - 8 *k **2 + 8))
    s_2 = (1/(2*lam*(lam*(1+k**2) - 2*k))) * ((3/2)*c_3*(2*a_22*(k**2 -2) + a_24*(k**2 +2) + 2*k*b_22 + 5*d_21) + 3/8 * c_4 * (12-k**2))
    l_1 = a_1 + 2 *(lam **2) * s_1
    l_2 = a_2 + 2 *(lam **2) * s_2
    
    #print(-delta)
    #print(- l_1 * A_x **2)
   #print((delta/l_2)**0.5)
    
    
    # EQ 18
    #x amplitude
    A_x = ((-delta - l_2 * A_z **2) / l_1)**(0.5)
    #z amplitude
    #A_z = ((-delta - l_1 * A_x **2) / l_2)**(0.5)
    #print("gamma \t ",gamma_L)    
    #print("lambda \t ",lam)
    #print("k \t ",k)
    #print("delta \t",delta)
    #print()

    
    #print(A_z * (gamma_L *a_1))
    
    #EQ 17
    v_2 = s_1 * A_x **2 + s_2 * A_z **2
    #tau = v*s
    v = 1 + v_2
    
    dt = (np.abs(lam)*v)
    
    #functions that calculate position in the orbit from the oribial phase from 0 to 2 pi
    x = lambda tau: a_21 * A_x ** 2   + a_22 * A_z  ** 2 - A_x  * np.cos(tau) + (a_23 * A_x ** 2 - a_24 * A_z **2 )* np.cos(2* tau) + (a_31* A_x **3 - a_32*A_x*A_z **2 ) * np.cos(3 * tau )
    y = lambda tau: k * A_x * np.sin(tau) + (b_21*  A_x **2 - b_22*A_z ** 2) * np.sin(2 *tau)+ (b_31 * A_x **2 - b_32 *A_x* A_z **2 )* np.sin(3*tau)
    z = lambda tau: delta_n * A_x * np.cos(tau) + delta_n * d_21* A_x* A_z* (np.cos( 2* tau) - 3) + delta_n *(d_32 * A_x**2  * A_z - d_31* A_z ** 2 )*  np.cos(3* tau)
    dx = lambda tau:  dt * (A_x  * np.sin(tau) - 2* (a_23 * A_x ** 2 - a_24 * A_z **2 )* np.sin(2* tau) - 3 * (a_31* A_x **3 - a_32*A_x*A_z **2 ) * np.sin(3 * tau ))
    dy = lambda tau: dt*(k * A_x * np.cos(tau) + 2*(b_21*  A_x **2 - b_22*A_z ** 2) * np.cos(2 *tau)+ 3*(b_31 * A_x **2 - b_32 *A_x* A_z **2 )* np.cos(3*tau))
    dz = lambda tau: dt*(- delta_n * A_x * np.sin(tau) + delta_n * -2 * d_21* A_x* A_z* np.sin( 2* tau)  - 3*delta_n *(d_32 * A_x**2  * A_z - d_31* A_z ** 2 )*  np.sin(3* tau))
    
    #period of the orbit
    p = 2*np.pi/(np.abs(lam)*v)
    return [p,x,y,z,dx,dy,dz]