#!/usr/bin/env python
# coding: utf-8
# A simple wrapper to make running experiment more convient
#
# Author : Julien Audiffren
# License : MIT License
# please cite the paper : "ZOOM: A Fast and Robust Solution to the Threshold Estimation Problem" if you use this algorithm for your research!


from responsefunction import ResponseF


import numpy as np
from tqdm import tqdm
from numpy import random




class XP_Wrapper():

    def __init__(self,response_function : ResponseF, optimization_method = None,
                 mu_star : float = 0.5, T : int= 1000,   repeats : int = 500) -> None:
        self.psi =response_function
        self.optim_method = optimization_method
        self.T = T
        self.mu_star = mu_star

        self.stimulus_target = None

        self.repeats = repeats 
        self._precompute()


    def _precompute(self):
        self.stimulus_target = self.psi.get_stimulus()
    
        
    def set_optim_method(self,optim):
        self.optim_method = optim


    def _loop(self):
        
        optim = self.optim_method(mu_star=self.mu_star, T=self.T)
        
        for _ in np.arange(1,self.T+1):           
            
            point_to_sample = optim.choose_arm()
            r  = random.rand()
            p  = self.psi.func(point_to_sample)
            observation = (r<p)
            optim.update_arm(int(observation))
           
        
        stimulus_estimator = optim.return_arm()

        regret = np.abs(self.psi.func(stimulus_estimator) - self.mu_star)

        return regret
                
            

    def run(self):
        
        results = []
        for _ in range(self.repeats) :
            results.append(self._loop())
        return results



                                      
                    
             