#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A class to simulate a response function
#
# Author : Julien Audiffren
# License : MIT License
# please cite the paper : "ZOOM: A Fast and Robust Solution to the Threshold Estimation Problem" if you use this algorithm for your research!



import numpy as np
import os






class ResponseF(): #generic class of psychometric function
    def __init__(self, type="LINEAR", delta=0.1,alpha = 1, mu_star=0.5,s_star=0.5):

        self.type = type
        self.delta=delta
        self.mu_star = mu_star
        self.s_star=s_star
        self.alpha = alpha

    def func(self,s):

        

        if self.type == "NONDIF":
            d = np.abs(s-self.s_star)
            if s> self.s_star:
                p =  self.mu_star +  np.power(d,2*self.alpha)
                
            else :
                p =  self.mu_star -  np.power(d,self.alpha)
        
        elif self.type == "LINEAR":
            d = (s-self.s_star)
            p =  self.mu_star +  d * self.alpha 

        elif self.type == "GAUSSIAN":

            p = stats.norm.cdf(x=s, loc=self.s_star, scale = self.delta) -0.5 + self.mu_star
                    
        p = np.minimum(np.maximum(p,self.delta),1.-self.delta)
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

