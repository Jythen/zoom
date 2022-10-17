# Author : Julien Audiffren
# License : MIT License
# please cite the paper : "ZOOM: A Fast and Robust Solution to the Threshold Estimation Problem" if you use this algorithm for your research!

import numpy as np

def compute_kl_dist(hmu : float,mu : float) -> float:
    """ 
        Function to compute the KL distance between the empirical average and mu_star

        Input

        hmu : the empirical average
        mu : the objective mu_star 
        
        Output :
        the distance

        """
    if mu == hmu :
        return 0
    if mu > hmu :
        if hmu >0 : 
            return  hmu*np.log(hmu/mu) + (1-hmu)*np.log((1-hmu)/(1-mu))
        else : 
            return - np.log((1-mu))
    else : 
        if hmu <1 : 
            return  hmu*np.log(hmu/mu) + (1-hmu)*np.log((1-hmu)/(1-mu))
        else : 
            return  - np.log((mu))
    

STR_ACTION_SAMPLE = "sample"
STR_ACTION_ZOOM = "zoom"


class OOM(): # Like ZOOM, but on ONE grid
    def __init__(self, mu_star : float, mininterval : float = 0., maxinterval : float = 1., T : int = 1000, K : int=8,  **kwargs): 
        """ 
        Class that implement the Find_Best_Interval and Choose_Interval_Extremity for a single grid.
        ZOOM is then build as a tree of grids.

        Parameters

        mu_star :  the desired output (float)
        mininterval : the minimun input value of the current grid. (float)
        maxinterval : the maximum input value of the current grid. (float)
        T : The total sampling budget
        K : The coarseness of the grid

        """
    
        

        self.T=T
        self.mu_star =  mu_star
        self.K = K
        
        self.mininterval = mininterval 
        self.maxinterval = maxinterval
        
        self.arm_dict = { k: [0,0,0,0] for k in range(1,self.K)}  # the dictionnary that stores for each arm k of the grid : Number of positive observation, number of total sample, relation according to (7), relation according to (8)
        self.arm_dict[0]=[0,0,-2,-2]  # extremities never need to be sampled
        self.arm_dict[self.K]=[0,0,2,2]

        self.last_sampled_point = self.K//2 #begins at the middle of the grid
        self.reverse_converter =  self._create_converter()  # functon to map the grid index k to its input value s_d,n,k
        
        
           
    def _create_converter(self):  
        """ Function that create the converter, ie. function to map the grid index k to its input value s_d,n,k """
        mininterval=self.mininterval
        maxinterval=self.maxinterval
        interval_size = (maxinterval - mininterval)/self.K      
        converter =   lambda k : k*interval_size + mininterval  
        return converter




    def _compute_CI_for_points(self,positive_obs : int, total_obs : int):
        """ 
        Compute the relation between the estimator of an arm value and the objective mu_star for both conditions

        Input

        positive_obs : the number of positive response 
        total_obs : total number of sample
        
        Output :
        estimator : the real value estimator of this arm probability
        relation_CI : if the arm is significantly smaller or bigger that s_*, according to the Azuma-Hoeffding CI. -2 means << , -1 <,0 no information yet, 1 >, 2 >> 
        relation_KL : if the arm is significantly smaller or bigger that s_*, according to the MOSS inequality. -2 means << , -1 <,0 no information yet, 1 >, 2 >> 


        """
            
        
        estimator =positive_obs/total_obs

        #compute relation with respect to the CI decision rule
       
        if total_obs ==0 :
            b= np.infty
            kl_error = np.infty
        else :
            b = np.sqrt(np.log(self.T)/total_obs)
            kl_error = np.log(self.T/(total_obs))/total_obs

        if estimator+b < self.mu_star :
            relation_CI = -2
        elif estimator -b > self.mu_star :
            relation_CI = 2
        elif estimator > self.mu_star :
            relation_CI = 1
        else :
            relation_CI = -1


        #compute relation with respect to the MOSS-KL decision rule

        kl_error = np.log(self.T/(total_obs))/total_obs
        d = compute_kl_dist(hmu=estimator,mu=self.mu_star)
        if estimator > self.mu_star:
            if d >= kl_error:
                relation_KL = 2
            else : 
                relation_KL = 1
        else : 
            if d >= kl_error:
                relation_KL = -2
            else : 
                relation_KL = -1
        
        return estimator, relation_CI, relation_KL
        


    

    def _compute_point_relation(self,k : int): # -2 means << , -1 <,0 no information yet, 1 >, 2 >> 
        """ 
        Compute the relation between the estimator of an arm value and the objective mu_star for both conditions

        Input

        k : the index of the arm 
                
        Output :
        relation_CI : if the arm is significantly smaller or bigger that s_*, according to the Azuma-Hoeffding CI. -2 means << , -1 <,0 no information yet, 1 >, 2 >> 
        relation_KL : if the arm is significantly smaller or bigger that s_*, according to the MOSS inequality. -2 means << , -1 <,0 no information yet, 1 >, 2 >> 


        """
        assert (k>0) & (k<self.K) # should not be called on extremities, as the relation is automatically << and >>
        
        stim_dict = self.arm_dict

        positive_pulls, total_pulls, _, _ = stim_dict[k]

        assert total_pulls>0,"called update on new point" # should only be called after arm is pulled
        
        _, relation_CH, relation_KL = self._compute_CI_for_points(positive_obs=positive_pulls,total_obs=total_pulls)
      
        return relation_CH,relation_KL



    def what_to_do(self, optimistic : bool = False) : 
        """ 
        Find what is the interval of interest for this grid, and whether the agent should zoom

        Input

        optimistic : Whether the decision rule used should be the MOSS inequality (optimistic=True) or the Azuma-Hoeffding CI (optimistic=False)
                
        Output :
        action : Whether to ZOOM or to Sample
        index : the interval of interest index

        """
        p0,t0,_,_ = self.arm_dict[self.last_sampled_point]
        if t0 == 0:
            action, index = STR_ACTION_SAMPLE, self.last_sampled_point
        else :
            ratio0 = p0/t0 
            if ratio0 > self.mu_star :
                high_idx = self.last_sampled_point
                low_idx = self.last_sampled_point-1
            else :
                high_idx = self.last_sampled_point +1
                low_idx = self.last_sampled_point

            _,_,rch0,rkl0 = self.arm_dict[low_idx]
            _,t1,rch1,rkl1 = self.arm_dict[high_idx]

            if optimistic : 
                r0, r1 = rkl0,rkl1
            else :
                r0, r1 = rch0,rch1

            if r0 == -2 :
                if r1 == 2 :
                    action, index = STR_ACTION_ZOOM, low_idx
                else :
                    action, index = STR_ACTION_SAMPLE, high_idx
            else :
                if r1 == 2 :
                    action, index = STR_ACTION_SAMPLE, low_idx
                elif t1 == 0 : 
                    action, index = STR_ACTION_SAMPLE, high_idx
                else :
                    action = STR_ACTION_SAMPLE
                    index = np.random.choice([low_idx,high_idx])
        if action == STR_ACTION_SAMPLE:
            self.last_sampled_point = index
        return action, index

    def zoom(self,index : int):
        """
        ZOOM on the given interval. Return a new instance of OOM


        Input

        index : the index of the interval of interest
                
        Output :
        a new OOM for this interval

        """

        
        new_interval_length = (self.maxinterval - self.mininterval)/self.K
        new_min = self.mininterval+index*new_interval_length
        new_max = new_min + new_interval_length
        return OOM(
                mu_star=self.mu_star,
                mininterval=new_min,
                maxinterval=new_max,
                T= self.T,
                K=self.K
            )

                
    def update_arm(self,answer : bool) -> None:
        """ 
        Update the arm estimator using an answer, i.e. whether the response function outputs 1 or 0

        Input

        answer, i.e. whether the response function outputs 1 or 0
                
        Output :
        None

        """
      
        index = self.last_sampled_point
        self.arm_dict[index][1]+=1
        self.arm_dict[index][0]+=int(answer)
        nrch,nrkl = self._compute_point_relation(index)
        self.arm_dict[index][2] = nrch
        self.arm_dict[index][3] = nrkl



    def get_most_pulled_arm(self):
        """ 
        Find which arm of the grid was pulled the most. Useful for final prediction

        Input

        None
                
        Output :
        the value of arm that was pulled the most
        the number of pull

        """
        nmax=0
        mindex=self.last_sampled_point
        for index in self.arm_dict :
            _,n,_,_ = self.arm_dict[index]
            if n>nmax :
                nmax= n
                mindex = index
        
        return self.reverse_converter(mindex), nmax

      



class ZOOM(): 
    """ 
    Class that implement the ZOOM Method, using a tree of OOM 

    Parameters

    mu_star :  the desired output (float)
    mininterval : the minimun input value of the current grid. (float)
    maxinterval : the maximum input value of the current grid. (float)
    T : The total sampling budget
    K : The coarseness of the grid

    """
    def __init__(self, mu_star : float, mininterval  : float = 0, maxinterval: float = 1, T : int = 1000, K : int=32, **kwargs): 
        """
        As DOS is completely model free, there are only 2 arguments
        mu*(the target proba), and T, the time horizon
        """

        

        self.T=T
        self.mu_star =  mu_star
        self.K = K
        

        self.mininterval = mininterval
        self.maxinterval = maxinterval

        self.tree_of_OOM = {
            "OOM" : OOM(
                mu_star=self.mu_star,
                mininterval=self.mininterval,
                maxinterval=self.maxinterval,
                T= self.T,
                K=self.K
            )
        }
        

        self.last_path = []  # store where the OOM used to sample was to propagate the update
        self.pull_optimistic = False  # store which rule should be used 
        self.N_star = T/(np.log(T)*np.log(np.log(T)))

        
        


        

    def choose_arm(self):
        """ 
        Find what arm should be sampled next

        Input
        None 


                
        Output :
        the value of the arm to sample 
        

        """ 
        dict = self.tree_of_OOM
        path = []
        action = None
        index = None

        while True:
            oom : OOM = dict["OOM"]
            action, index = oom.what_to_do(optimistic=self.pull_optimistic)
            if action == STR_ACTION_SAMPLE :
                break
            else :
                path.append(index)
                if not (index in dict) :
                    new_oom = oom.zoom(index=index)
                    dict[index] = { "OOM" : new_oom}
                dict = dict[index]
        
        value = oom.reverse_converter(k=index)
        self.last_path = path
        return value
                    



    def update_arm(self,answer : bool):
        """ 
        Use the answer to update the arm value in the correct OOM

        Input

        answer, i.e. whether the response function outputs 1 or 0
                
        Output :
        None

        """

        dict = self.tree_of_OOM
        path = self.last_path
        

        for p in path : 
            dict = dict[p]
        oom : OOM = dict["OOM"]
        oom.update_arm(answer=answer)

        self.pull_optimistic = not self.pull_optimistic
            

    def return_arm(self):
        """ 
        compute the final prediction

        Input

        None
                
        Output :
        the value of the final prediction

        """

        dict = self.tree_of_OOM
        
        action = None
        index = None
        promising_arm = None
        promising_arm_pull = 0
        

        while True:
            oom : OOM = dict["OOM"]

            value, npulls = oom.get_most_pulled_arm()
            if npulls > self.N_star :
               if npulls > promising_arm_pull:
                promising_arm_pull = npulls
                promising_arm = value
            
            action, index = oom.what_to_do(optimistic=True)
            if action == STR_ACTION_SAMPLE :
                break
            elif not (index in dict) :
                break
            else :
                dict = dict[index]
        
        if not( promising_arm is None ):
            return promising_arm
        else :
            value, npulls = oom.get_most_pulled_arm()
            return value


