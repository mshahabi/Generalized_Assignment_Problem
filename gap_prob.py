# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:47:01 2018

@author: mshahabi
"""

"""
Mathematical formulation for generalized assgnement problem 

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
    
    def gen_rand_problem(nbM,nbJ):
        cost_ = np.empty([nbM,nbJ], dtype=float)
        cap = np.empty([nbM], dtype=float)
        for M in range(0, nbM):
            cap[M] = 3
            for J in range(0, nbJ):
                 cost_[M,J] = random.randint(0,100)       
        return cost_, cap    

