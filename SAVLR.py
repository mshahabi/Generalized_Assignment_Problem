# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:32:57 2018

@author: mshahabi
"""
import numpy as np
from gap_prob import Central_GAP, SLR_SubProblem
from config_opt import config


class Initial_SAVLR:
    
    def __init__(self, config):
        
        self.relaxed_nbM = config["num_of_machines"]
        if config["genrate_new_problem"]==True:
            self.cost, self.cap = SLR_SubProblem.gen_rand_problem(config["num_of_machines"],config["num_of_jobs"])
        self.relaxed_cap = self.cap*2    
        self.lambdaa = [-12 for i in range(0, config["num_of_jobs"])] 
        self.c_k_old = 0
        self.s_k = 0.2
        
    def solve_problems(self):
        ################EXACT RESULTS###############################################
        gen_exact_problem = Central_GAP(config["num_of_machines"], config["num_of_jobs"], self.cost, self.cap)
        self.q_e, self.x_e = gen_exact_problem.solve(config["display_solution_stream"],config["relax_solution"])  

        ############SOLVING A RELAXED PROBLEM TO GET q_0###########################
        gen_relax_problem = Central_GAP(self.relaxed_nbM, config["num_of_jobs"], self.cost, self.relaxed_cap)
        self.q_0, x_0 = gen_relax_problem.solve(config["display_solution_stream"],config["relax_solution"])

        ####################SOLVING A RELAXED PROBLEM TO GET Lagrangian function#######################  
        #Solving the relaxed problem to get Lagrangian function
        gen_lagrangian = SLR_SubProblem(self.lambdaa,self.s_k, config["num_of_machines"], config["num_of_jobs"], self.cost, self.cap)
        self.Lagrang, self.x_0, self.q_ = gen_lagrangian.solve_sp(-1, x_0, False, False)

        Lagrang_k = sum(x_0[m,j]*(self.cost[m,j] + self.lambdaa[j]) for m in range(0, config["num_of_machines"]) for j in range(0, config["num_of_jobs"]))-sum(self.lambdaa[j] for j in range(0,config["num_of_jobs"])) + sum(self.q_[j]*0.5*self.s_k for j in range(0, config["num_of_jobs"]))
    
    def initialize(self):
        
        self.solve_problems()
        self.g_m = np.empty([config["num_of_jobs"]], dtype=int)
        for j in range(0, config["num_of_jobs"]):
            self.g_m[j] =  self.x_0[:,j].sum()-1
        self.g_m_old = sum(self.g_m**2)  
          
        if self.g_m_old !=0:   
           self.c_k_old = (self.q_0-self.Lagrang)/self.g_m_old
#           self.lambdaa = self.lambdaa + self.c_k_old*self.g_m
        if self.g_m_old ==0 and self.c_k_old==0:
           print("Initial Solution is Feasable")
           self.obj = sum(self.x_0[m,j]*self.cost[m,j] for m in range(0, config["num_of_machines"]) for j in range(0, config["num_of_jobs"]))


class SAVLR(Initial_SAVLR):
    
    def __init__(self, config):
        
        Initial_SAVLR.__init__(self, config)
        Initial_SAVLR.initialize(self)
        self.sub_sol = np.empty([config["num_of_machines"], config["num_of_jobs"]], dtype=int)
        
    def start_savlr(self):     
        
        print("***********STARTING SAVLR ALGORITHM*************")
        
        NumOfSub = config["num_of_sub_problems"]
        sub_interval = int(config["num_of_machines"]/NumOfSub)
        sub_number = 0 
        sub_index = 0
        max_penalty = 2.0
        
        for k in range(1, config["ItrNum"]):
            
            print("**************************************")   
            print("Iteration %s has been started"%k, "S_k is equal to %s"%self.s_k)
            print("**************************************")                                   
                
            if sub_number<config["num_of_sub_problems"]:
                sub_number = sub_number + 1
            else:
                sub_number = 1
                sub_index = 0 
            next_sub_index = sub_number*sub_interval
            sub_problem = [i for i in range(sub_index, next_sub_index)]
            sub_index = sub_number*sub_interval
             
            
            print(sub_problem)
            print("Solving solving the %s sub problem"%sub_number)
            
            sp = SLR_SubProblem(self.lambdaa, self.s_k, config["num_of_machines"], config["num_of_jobs"], self.cost, self.cap)
            self.Lagrang_sp, self.x_sp, self.q_sp = sp.solve_sp(sub_problem, self.x_0, False, False)
            Lagrang_k = sum(self.x_0[m,j]*(self.cost[m,j] + self.lambdaa[j]) for m in range(0,config["num_of_machines"]) for j in range(0, config["num_of_jobs"]))-sum(self.lambdaa[j] for j in range(0,config["num_of_jobs"])) + sum(self.q_[j]*0.5*self.s_k for j in range(0, config["num_of_jobs"]))
            
           
            if int(round(self.Lagrang_sp))<=int(round(Lagrang_k)):                
                print("Sub Gredient Optimality Condition is acheived")
                for j in range(0, config["num_of_jobs"]):
                    self.g_m[j] =  self.x_sp[:,j].sum() - 1
                self.g_m_new = sum(self.g_m**2)    
                
                if self.g_m_new !=0:
                    r = 1/k**0.5 + 0.01
#                    (1-1/config["M"]/(k**(1-1/k**r)))
                    step = 0.96*(self.c_k_old+0.0000001)*((self.g_m_old+0.0000001)/(self.g_m_new+0.0000001))
                    if(k> config["num_of_machines"]/config["num_of_jobs"] and k<config["num_of_machines"]*10/config["num_of_jobs"]): 
                        step = config["num_of_sub_problems"]/config["num_of_machines"]*(self.q_0-self.Lagrang_sp)/self.g_m_new ;
#                    p = 1 - 1/(k**config["r"]) + 0.01
#                    alpha = 1- 1/(config["M"]*k**p)
#                    print("p",p, alpha)
                    self.c_k = 0.99*(self.q_0-self.Lagrang_sp)*self.g_m_old/self.g_m_new
                    print("Lambdaa", self.lambdaa)
                    print("Change in Lambda", step*self.g_m)
                    self.lambdaa = self.lambdaa + step*self.g_m
                    print("New Lambda", self.lambdaa)
                    self.g_m_old = self.g_m_new
                    self.c_k_old = step
                    self.obj = sum(self.x_sp[m,j]*self.cost[m,j] for m in range(0, config["num_of_machines"]) for j in range(0, config["num_of_jobs"]))
                    self.x_0 = self.x_sp
                    self.q_  = self.q_sp
                    if self.s_k < max_penalty:
                        self.s_k = self.s_k*1.01
                    print("**************************************")   
                    print("**************************************")
                    print("Error Rate", sum(sum((self.x_e-self.x_sp)**2)))
                    print("GAP Rate", (self.obj-self.q_e)/self.obj)   
                    print("**************************************")
                    print("**************************************")
                    if sum(sum((self.x_e-self.x_sp)**2)) <=3: break
                else:
                    self.obj = sum(self.x_sp[m,j]*self.cost[m,j] for m in range(0, config["num_of_machines"]) for j in range(0, config["num_of_jobs"]))    
                    print("Exact Obj","Relaxed","Lagrang_sp","Lagrang_k")    
                    print(self.q_e, ",,,," , self.obj, ",,," ,   self.Lagrang_sp, "," , Lagrang_k)
                    self.s_k = self.s_k/1.02  
                    max_penalty = self.s_k
            else:
                 print("Sub Gredient Optimality Condition is NOT acheived")
                 max_penalty = self.s_k
                 self.s_k = self.s_k/1.02
                 
                 
        return self.x_e, self.x_0
a = SAVLR(config)
x_e, x_p = a.start_savlr()

