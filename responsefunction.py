#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A class to simulate a response function



import numpy as np
import os







class ResponseF(): #generic class of psychometric function
    def __init__(self, delta=0.1,alpha = 1, mu_star=0.5,s_star=0.5):

        self.delta=delta
        self.mu_star = mu_star
        self.s_star=s_star
        self.alpha = alpha
        

        self._random_pulls = []
        self._random_index = 0

    
    def func(self,s):
    
        if s> self.s_star:
            d = min(s-self.s_star, self.delta)
            p =  self.mu_star +  np.power(d,self.alpha)
            
        else :
            d = min(self.s_star-s, self.delta)
            p =  self.mu_star -  np.power(d,self.alpha)
        
        p = np.minimum(np.maximum(p,0.01),0.99)
        return  p


    def get_stimulus(self,precision=1e-3):
        
        x_space = np.arange(precision,1,precision)
        y_space = [self.func(x) for x in x_space]
        z_space = np.abs(self.mu_star - np.array(y_space))
        z = np.argmin(z_space)
        return(x_space[z])
    


    def sample(self, stimulus):
        proba = self.func(stimulus)
        r = np.random.random(1)
        return (proba>r)

    def reset_s_star(self):
        self.s_star = np.random.uniform(low=0.15,high=0.85)




