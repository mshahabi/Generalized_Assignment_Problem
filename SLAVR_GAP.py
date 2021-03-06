# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:47:01 2018

@author: mshahabi
"""

"""
Solving a generalized assgnement problem by surrogate abselute value lagrangian relaxation

"""

import os
import random
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from scipy import sparse
import math
import logging
import numpy as np
import cplex

class Central_GAP:
    
    def __init__(self,nbM,nbJ,cost,capacity):
        self.nbM = nbM
        self.nbJ = nbJ
        self.cost, self.b = cost, capacity
        self.Create_Model()
           
    def Create_Model(self):
        self.model = ConcreteModel()
        self.model.nbM   = Set(initialize=range(0,self.nbM))
        self.model.nbJ   = Set(initialize=range(0,self.nbJ))
        self.model.cost    = self.cost
        self.model.b       = self.b 
        self.model.x       = Var(self.model.nbM*self.model.nbJ, within=Binary)   
        
        def obj_rule(model):
            first_term  = sum(self.model.x[n]*self.model.cost[n] for n in self.model.nbM*self.model.nbJ)          
            return first_term
            
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
            
        def flow_balance_1(model,m):
            return sum(self.model.x[m,j] for j in self.model.nbJ)<= self.model.b[m] 
        
        def flow_balance_2(model,j):          
            return sum(self.model.x[m,j] for m in self.model.nbM)==1 
                       
        self.model.flowbal_1 = Constraint(self.model.nbM, rule=flow_balance_1)
        self.model.flowbal_2 = Constraint(self.model.nbJ, rule=flow_balance_2)
            
    def solve(self, display_solution_stream=False , solve_relaxation = False):
        instance = self.model
        instance.preprocess()
        opt = SolverFactory("cplex")
        opt.relax = solve_relaxation
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        x_v = np.empty([self.nbM,self.nbJ], dtype=int) 
        for p in instance.nbM:
            for q in instance.nbJ:
                    x_v[p,q] = instance.x[p,q].value
        return value(instance.obj), x_v 

class SLR_SubProblem():
    
    def __init__(self, lambdaa, s_k, nbM, nbJ, cost, capacity):
        self.cost, self.b = cost, capacity
        self.nbM = nbM
        self.nbJ = nbJ
        self.s_k = s_k
        self.Create_Sub_Problem(lambdaa)
          
    def Create_Sub_Problem(self, lambdaa):
        self.model         = ConcreteModel()
        self.lambdaa       = lambdaa
        self.model.nbM     = Set(initialize=range(0,self.nbM))
        self.model.nbJ     = Set(initialize=range(0,self.nbJ))
        self.model.cost    = self.cost
        self.model.b       = self.b
        self.model.x       = Var(self.model.nbM*self.model.nbJ, within=Binary)
        self.model.q      = Var(self.model.nbJ, within=NonNegativeReals)

        def obj_rule(model):
            first_term  = sum(self.model.x[n]*(self.model.cost[n]+self.lambdaa[n[1]]) for n in self.model.nbM*self.model.nbJ)
            second_term = sum(self.model.q[j]*0.5*self.s_k for j in self.model.nbJ) -  sum(lambdaa[j] for j in self.model.nbJ) 
            return first_term + second_term            
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
            
        def flow_balance_1(model,m):
            return sum(self.model.x[m,j] for j in self.model.nbJ)<= self.model.b[m] 
        
        def flow_balance_2(model,j):          
            return  sum(self.model.x[m,j] for m in self.model.nbM)-1 <= self.model.q[j]
        
        def flow_balance_3(model,j):          
            return  -self.model.q[j] <= sum(self.model.x[m,j] for m in self.model.nbM)-1
                       
        self.model.flowbal_1 = Constraint(self.model.nbM, rule=flow_balance_1)            
        self.model.flowbal_2 = Constraint(self.model.nbJ, rule=flow_balance_2)  
        self.model.flowbal_3 = Constraint(self.model.nbJ, rule=flow_balance_3)  
        
    def solve_sp(self, solve_for_machine, x_f, display_solution_stream=False , solve_relaxation = False):
        instance = self.model
        if solve_for_machine != -1:
            for m in range(0, self.nbM):
                if m != solve_for_machine:
                    for j in range(0, self.nbJ):
                       instance.x[m,j] = x_f[m,j]
                       instance.x[m,j].fixed = True
        instance.preprocess()
        opt = SolverFactory("cplex" )
        opt.relax = solve_relaxation
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        x_s = np.empty([self.nbM,self.nbJ], dtype=int)
        q_s = np.empty([self.nbJ], dtype=float) 
        for p in instance.nbM:
            for q in instance.nbJ:
                    x_s[p,q] = instance.x[p,q].value
        for j in instance.nbJ:
                    q_s[j] = instance.q[j].value            
        return value(instance.obj), x_s, q_s 
    
#########################Function to generate a random problem#################
def gen_rand_problem(nbM,nbJ):
     cost_ = np.empty([nbM,nbJ], dtype=float)
     cap = np.empty([nbM], dtype=float)
     for M in range(0, nbM):
         cap[M] = 2
         for J in range(0, nbJ):
             cost_[M,J] = random.randint(0,100)       
     return cost_, cap    
################### SET THE SOLVER#############################################
solver_name =  "cplex"
display_solver_log = False
genrate_new_problem = False
relax_solution = False
num_of_machines = 10
num_of_jobs = 12
relaxed_nbM = num_of_machines
if genrate_new_problem==True:
    cost,cap = gen_rand_problem(num_of_machines,num_of_jobs)
relaxed_cap = cap*2
 #Initializing Lambda0
lambda_acum = {"x": [], "y": []}
lambdaa = [-100 for i in range(0,num_of_jobs)] 
lambda_acum["x"].append(lambdaa[0])
lambda_acum["y"].append(lambdaa[1])
#SLR initial paramters
alpha = 0.06
M = 10
r = 0.5
ItrNum = 28

#####################EXACT RESULTS###############################################
gen_exact_problem = Central_GAP(num_of_machines, num_of_jobs, cost, cap)

q_e, x_e = gen_exact_problem.solve(display_solver_log,relax_solution)  

####################SOLVING A RELAXED PROBLEM TO GET q_0#######################

gen_relax_problem = Central_GAP(relaxed_nbM, num_of_jobs, cost, relaxed_cap)

q_0, x_0 = gen_relax_problem.solve(display_solver_log,relax_solution)
####################SOLVING A RELAXED PROBLEM TO GET Lagrangian function#######################  
#Solving the relaxed problem to get Lagrangian function
c_k_old = 0
s_k = 65
counter_ = 1
gen_lagrangian = SLR_SubProblem(lambdaa,s_k, num_of_machines, num_of_jobs, cost, cap)
Lagrang, x_0, q_ = gen_lagrangian.solve_sp(-1, x_0, False, False)

Lagrang_k = sum(x_0[m,j]*(cost[m,j] + lambdaa[j]) for m in range(0,num_of_machines) for j in range(0,num_of_jobs))-sum(lambdaa[j] for j in range(0,num_of_jobs)) + sum(q_[j]*0.5*s_k for j in range(0,num_of_jobs))
Lagrang_sp = Lagrang_k 
g_m = np.empty([num_of_jobs], dtype=int)
for j in range(0,num_of_jobs):
    g_m[j] =  x_0[:,j].sum()-1
g_m_old = sum(g_m**2)  
  
if g_m_old !=0:   
   c_k_old = (q_0-Lagrang)/g_m_old
   lambdaa = lambdaa + c_k_old*g_m
else:
    print("Optimality is acheived no need to continue")
    ItrNum = 1
    obj = sum(x_0[m,j]*cost[m,j] for m in range(0,num_of_machines) for j in range(0,num_of_jobs))

sub_sol = np.empty([num_of_machines, num_of_jobs], dtype=int) 
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
print(q_e, ",,,," , obj, ",,," ,   Lagrang_sp, "," , Lagrang_k)    