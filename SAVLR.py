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
        self.lambdaa = [-100 for i in range(0, config["num_of_jobs"])] 
        self.c_k_old = 0
        self.s_k = 65
        
    def solve_problems(self):
        ################EXACT RESULTS###############################################
        gen_exact_problem = Central_GAP(config["num_of_machines"], config["num_of_jobs"], self.cost, self.cap)
        self.q_e, self.x_e = gen_exact_problem.solve(config["display_solver_log"],config["relax_solution"])  

        ############SOLVING A RELAXED PROBLEM TO GET q_0###########################
        gen_relax_problem = Central_GAP(self.relaxed_nbM, config["num_of_jobs"], self.cost, self.relaxed_cap)
        self.q_0, x_0 = gen_relax_problem.solve(config["display_solver_log"],config["relax_solution"])

        ####################SOLVING A RELAXED PROBLEM TO GET Lagrangian function#######################  
        #Solving the relaxed problem to get Lagrangian function
        gen_lagrangian = SLR_SubProblem(self.lambdaa,self.s_k, config["num_of_machines"], config["num_of_jobs"], self.cost, self.cap)
        self.Lagrang, self.x_0, self.q_ = gen_lagrangian.solve_sp(-1, x_0, False, False)

        Lagrang_k = sum(x_0[m,j]*(self.cost[m,j] + self.lambdaa[j]) for m in range(0, config["num_of_machines"]) for j in range(0, config["num_of_jobs"]))-sum(self.lambdaa[j] for j in range(0,config["num_of_jobs"])) + sum(self.q_[j]*0.5*self.s_k for j in range(0, config["num_of_jobs"]))
        Lagrang_sp = Lagrang_k 
    
    def initialize(self):
        
        self.solve_problems()
        self.g_m = np.empty([config["num_of_jobs"]], dtype=int)
        for j in range(0, config["num_of_jobs"]):
            self.g_m[j] =  self.x_0[:,j].sum()-1
        self.g_m_old = sum(self.g_m**2)  
          
        if self.g_m_old !=0:   
           self.c_k_old = (self.q_0-self.Lagrang)/self.g_m_old
           self.lambdaa = self.lambdaa + self.c_k_old*self.g_m
        if self.g_m_old ==0 and self.c_k_old==0:
            print("Optimality is acheived no need to continue")
            self.obj = sum(self.x_0[m,j]*self.cost[m,j] for m in range(0, config["num_of_machines"]) for j in range(0, config["num_of_jobs"]))

start_=Initial_SAVLR(config)
start_.initialize()
"""
sub_sol = np.empty([config["num_of_machines"], config["num_of_jobs"]], dtype=int) 
sub_counter = 0

for k in range(1, ItrNum):
    print("**************************************")   
    print("Iteration %s has been started"%k)
    print("**************************************")
    sub_counter = 0
    flag = 0
    main_counter = 0 
    print("Solving the first sub problem")
    sp = SLR_SubProblem(lambdaa, s_k, num_of_machines, num_of_jobs, cost, cap)
    Lagrang_sp, x_sp, q_sp = sp.solve_sp(sub_counter, x_0, False, False)
    Lagrang_k = sum(x_0[m,j]*(cost[m,j] + lambdaa[j]) for m in range(0,num_of_machines) for j in range(0,num_of_jobs))-sum(lambdaa[j] for j in range(0,num_of_jobs)) + sum(q_[j]*0.5*s_k for j in range(0,num_of_jobs))
    print(Lagrang_sp, Lagrang_k )
    if Lagrang_sp>=Lagrang_k :
        while flag==0:
            print("Start Solving the Second Sub Problem")
            sp = SLR_SubProblem(lambdaa, s_k, num_of_machines, num_of_jobs, cost, cap)
            Lagrang_sp, x_sp, q_sp = sp.solve_sp(sub_counter, x_0, False, False)
            Lagrang_k = sum(x_0[m,j]*(cost[m,j] + lambdaa[j]) for m in range(0,num_of_machines) for j in range(0,num_of_jobs))-sum(lambdaa[j] for j in range(0,num_of_jobs)) + sum(q_[j]*0.5*s_k for j in range(0,num_of_jobs))
            sub_counter = sub_counter + 1
            main_counter = main_counter +1
            if sub_counter>=num_of_machines:
                sub_counter = 0
                s_k = s_k/2
            if Lagrang_sp<Lagrang_k:
                print("Lagrangian Optimality conditioned is achevied")
                flag = 1
                print(Lagrang_sp,Lagrang_k )
            if main_counter>20:
                print("Lagrangian Optimality conditioned is not found: Returning the previous solution")
                x_sp = x_0
                q_sp = q_
                flag = 1                
    for j in range(0,num_of_jobs):
        g_m[j] =  x_sp[:,j].sum()-1
    g_m_new = sum(g_m**2)    
    if g_m_new !=0:
        p = 1 - 1/(k**r)
        alpha = 1- 1/(M*k**p)
        c_k = alpha*c_k_old*g_m_old/g_m_new
        print("Lambdaa",lambdaa)
        print("Change in Lambda",c_k*g_m)
        lambdaa = lambdaa + c_k*g_m
        print("New Lambda", lambdaa)
        g_m_old = g_m_new
        c_k_old = c_k
        obj = sum(x_sp[m,j]*cost[m,j] for m in range(0,num_of_machines) for j in range(0,num_of_jobs))
        x_0 = x_sp
        q_  = q_sp
        s_k = s_k*1.1
        print("**************************************")   
        print("**************************************")
        print("Error Rate", sum(sum((x_e-x_sp)**2)))
        print("GAP Rate", (obj-q_e)/obj)   
        print("**************************************")
        print("**************************************")
        if (q_e-Lagrang_sp)/Lagrang_sp==0.00: break
    else:
        obj = sum(x_sp[m,j]*cost[m,j] for m in range(0,num_of_machines) for j in range(0,num_of_jobs))
        x_0 = x_sp
        q_  = q_sp
        s_k = s_k/1.1

print("Exact Obj","Relaxed","Lagrang_sp","Lagrang_k")    
print(q_e, ",,,," , obj, ",,," ,   Lagrang_sp, "," , Lagrang_k)    """