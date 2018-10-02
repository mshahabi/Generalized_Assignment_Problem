# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:47:01 2018

@author: mshahabi
"""

"""
Solving a generalized assgnement problem by surrogate lagrangian relaxation

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
            second_term = sum(self.s_k*self.model.q[j]*0.5 for j in self.model.nbJ) 
            return first_term - second_term            
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
            
        def flow_balance_1(model,m):
            return sum(self.model.x[m,j] for j in self.model.nbJ)<= self.model.b[m] 
        
        def flow_balance_2(model,j):          
            return  -self.model.q[j] <= sum(self.model.x[m,j] for m in self.model.nbM)-1 <= self.model.q[j]
                       
        self.model.flowbal_1 = Constraint(self.model.nbM, rule=flow_balance_1)            
        self.model.flowbal_2 = Constraint(self.model.nbJ, rule=flow_balance_2)    
        
    def solve_sp(self, display_solution_stream=False , solve_relaxation = False):
        instance = self.model
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
        for p in instance.nbM:
            for q in instance.nbJ:
                    x_s[p,q] = instance.x[p,q].value
        return value(instance.obj), x_s 
    
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
num_of_machines = 10
num_of_jobs = 12
relaxed_nbM = num_of_machines
cost,cap = gen_rand_problem(num_of_machines,num_of_jobs)
relaxed_cap = cap*3
 #Initializing Lambda0
lambda_acum = {"x": [], "y": []}
lambdaa = [-10 for i in range(0,num_of_jobs)] 
lambda_acum["x"].append(lambdaa[0])
lambda_acum["y"].append(lambdaa[1])
#SLR initial paramters
alpha = 0.6
M = 100
r = 0.5
ItrNum = 80
#################2#EXACT RESULTS###############################################
gen_exact_problem = Central_GAP(num_of_machines, num_of_jobs, cost, cap)
display_solver_log = False
relax_solution = False
q_e, x_e = gen_exact_problem.solve(display_solver_log,relax_solution)  
####################SOLVING A RELAXED PROBLEM TO GET q_0#######################
gen_relax_problem = Central_GAP(relaxed_nbM, num_of_jobs, cost, relaxed_cap)
display_solver_log = False
relax_solution = False
q_0, x_0 = gen_relax_problem.solve(display_solver_log,relax_solution)
####################SOLVING A RELAXED PROBLEM TO GET Lagrangian function#######################  
#Solving the relaxed problem to get Lagrangian function
c_k_old = 1
gen_lagrangian = SLR_SubProblem(c_k_old,lambdaa, num_of_machines, num_of_jobs, cost, cap)
display_solver_log = False
relax_solution = False
Lagrang, x_0 = gen_lagrangian.solve_sp(solver_name, False)

g_m = np.empty([num_of_jobs], dtype=int)
for j in range(0,num_of_jobs):
    g_m[j] =  x_0[:,j].sum()-1
g_m_old = sum(g_m**2)    
if g_m_old ==0:
    g_m_old = 0.001
c_k_old = (q_0-Lagrang)/g_m_old
lambdaa = lambdaa + c_k_old*g_m
sub_sol = np.empty([num_of_machines, num_of_jobs], dtype=int) 
for k in range(1, ItrNum):
    print(lambdaa, c_k_old*g_m )
    sp = SLR_SubProblem(c_k_old,lambdaa, num_of_machines, num_of_jobs, cost, cap)
    _, x_sp = sp.solve_sp(solver_name, False)

    for j in range(0,num_of_jobs):
        g_m[j] =  x_sp[:,j].sum()-1
    g_m_new = sum(g_m**2)    
    if g_m_new ==0:
        g_m_new = 0.001
    p = 1 - 1/(k**r)
    alpha = 1- 1/(M*k**p)
    c_k = alpha*c_k_old*g_m_old/g_m_new
    lambdaa = lambdaa + c_k*g_m

    g_m_old = g_m_new
    c_k_old = c_k
    obj = sum(x_sp[m,j]*cost[m,j] for m in range(0,num_of_machines) for j in range(0,num_of_jobs))

       
