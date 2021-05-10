#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:43:02 2021

@author: luke
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

e_m = 5.972 * 10**24
Is = 300
T = 120
G = 6.67408 * 10**(-11)
r_e = 6371 *10 ** 3
m_i = 320 
def start_thrust(t,r):
    x,y,dx,dy,m = r
    s = (x**2 + y**2)**(1/2)
    t_0 = y/s + 0.05
    return t_0
def end_thrust(t,r):
    x,y,dx,dy,m = r
    s = (x**2 + y**2)**(1/2)
    t_0 = y/s - 0.05
    return t_0

def eom(t,r):
    """2BP for photon"""

    x,y,dx,dy,m = r
    #print(t)
    dm = 0
    s = (x**2 + y**2)**(1/2)    
    #print(s)
    v_x =dx
    v_y =dy
    v_s = (dx ** 2 + dy ** 2) ** (1/2)
    a_x = -(G*e_m/(s**2)) * (x/s)
    a_y = -(G*e_m/(s**2)) * (y/s) 
    return [v_x,v_y,a_x,a_y,dm]

def eom_thrust(t,r):
    """2BP with Thrust from photon engine"""
    

    x,y,dx,dy,m = r
    dm = - T/(9.80665* Is)
    #print(t)
    s = (x**2 + y**2)**(1/2)    
    #print(s)
    v_x =dx
    v_y =dy
    v_s = (dx ** 2 + dy ** 2) ** (1/2)
    a_x = -(G*e_m/(s**2)) * (x/s) + T/m * (v_x/v_s)
    a_y = -(G*e_m/(s**2)) * (y/s)  + T/m * (v_y/v_s)
    
    #print(a_x)
    #print(m)
    return [v_x,v_y,a_x,a_y,dm]

start_thrust.terminal  = True
start_thrust.direction  = 1

end_thrust.terminal  = True
end_thrust.direction  = 1
    
r_o = r_e + 250*10**3
o_v = (G * e_m / (r_e + 250*10**3)) ** (1/2) 
o_v = o_v

t_f = ((4*np.pi ** 2)/(G*e_m) *  (r_o ** 3)) ** (1/2)
t_s=np.arange(0,1.5*t_f,t_f/100)
r_0 = [-(250*10**3 + r_e),0,0,o_v,m_i]

result = solve_ivp(eom,(0,t_f),r_0,dense_output=True,method='DOP853',first_step=1,events=start_thrust,max_step = 5)

t=np.arange(0,result.t_events[0][0],t_f/500)
pnts = result.sol(t)
x = pnts[0]
print((x[0] + r_e)/(10**3))
y = pnts[1]
plt.plot(x,y)
phi = np.arange(0,2*np.pi,2*np.pi/200)
x_e = (r_e) * np.cos(phi)
y_e = (r_e) * np.sin(phi)
plt.plot(x_e,y_e)

i = 0
while i < 100:
    
    r_0 = result.y_events[0][0]
    t_0 = result.t_events[0][0]
    result = solve_ivp(eom_thrust,(t_0,10*t_0),r_0,dense_output=True,method='DOP853',first_step=0.1,events=end_thrust,max_step = 1)
    
    
    t=np.arange(t_0,result.t_events[0][0],t_f/500)
    pnts = result.sol(t)
    
    x_t = pnts[0]
    y_t = pnts[1]
    m = pnts[4][-1]
    plt.plot(x_t,y_t)
    print((min(x_t)+ r_e)/(10**3))
    
    r_0 = result.y_events[0][0]
    t_0 = result.t_events[0][0]
    
    
    result = solve_ivp(eom,(t_0,10*t_0),r_0,dense_output=True,method='DOP853',first_step=t_0/5000,events=start_thrust,max_step = t_0/1000)
    
    try:
        t=np.arange(t_0,result.t_events[0][0],t_f/500)
        pnts = result.sol(t)
        x_r = pnts[0]
        y_r = pnts[1]
        plt.plot(x_r,y_r)
        i += 1
    except IndexError:
        print("Index_error")
        t=np.arange(t_0,1.1*t_0)
        pnts = result.sol(t)
        x_r = pnts[0]
        y_r = pnts[1]
        plt.plot(x_r,y_r,label="Escape")
        i = 100
    except:
        print("Something else went wrong")

print("time in days: ",t_0/86400)
print("Final Mass: ",m)
h = ((2*G*e_m) * (-min(x_t))) ** (1/2)
plt.title("Example orbit raising")
plt.xlabel("x axis (km)")
plt.ylabel("y axis (km)")
v = np.arange(0,np.pi,np.pi/1000)[:965]
r = ((h ** 2)/(G*e_m)) * 1 / (1+np.cos(v))
plt.title()
plt.plot(-r*np.cos(v),r*np.sin(v),label = "c3")
plt.legend()

