#!/usr/bin/env python
# coding: utf-8
#
# An example on how to use ZOOM
#
# Author : Julien Audiffren
# License : MIT License
# please cite the paper : "ZOOM: A Fast and Robust Solution to the Threshold Estimation Problem" if you use this algorithm for your research!





import numpy as np


from ZOOM import ZOOM
from responsefunction import ResponseF
from xp_wrapper import XP_Wrapper  
                    

 # xp params

mu_star = 0.2
repeats = 20
T = 50
alpha = 0.5
delta = 0.1


if __name__ == "__main__":

   
 


    print("using alpha = {}, delta = {}, mu_star = {}".format(alpha,delta,mu_star))
    

    
    psi = ResponseF(delta=delta,alpha=alpha,mu_star=mu_star)      
    psi.reset_s_star()

    xpwrap = XP_Wrapper(response_function=psi, optimization_method=ZOOM, \
        mu_star=mu_star, T=T,repeats=repeats)

    regrets = xpwrap.run()


    print("Achieved Regret : {}  (+- {} )".format(np.mean(regrets),np.std(regrets)))
    