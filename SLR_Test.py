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

class Central_Test:
    
    def __init__(self, number_of_x, a1, a2, obj_cost, variable_type):
        self.a1 = a1
        self.a2 = a2
        self.cost = obj_cost
        self.number_of_x = number_of_x
        self.Create_Model(variable_type)
       
   
    def Create_Model(self, variable_type):
        self.model = ConcreteModel()
        self.model.nodes   = Set(initialize=range(0, self.number_of_x))
        self.model.x       = Var(self.model.nodes, within = variable_type)   
        def obj_rule(model):
            first_term  = sum((self.model.x[n]**2)*self.cost[n] for n in self.model.nodes)          
            return first_term
            
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
            
        def flow_balance_1(model):
            return sum(self.model.x[n]*self.a1[n] for n in self.model.nodes)<=-48
        
        def flow_balance_2(model):          
            return sum(self.model.x[n]*self.a2[n] for n in self.model.nodes)<=-250 
        
                
        self.model.flowbal_1 = Constraint(rule=flow_balance_1)
        self.model.flowbal_2 = Constraint(rule=flow_balance_2)
        self.model.pprint    
        
    def solve(self, solver_name, display_solution_stream=False , solve_relaxation = False):
        instance = self.model
        opt = SolverFactory(solver_name)
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        x_v = [] 
        for p in instance.nodes:
                    x_v.append(instance.x[p].value)
        return value(instance.obj), np.array(x_v)



class Sub_Problem:
    
    def __init__(self, lambdaa, cost):
        self.lambdaa = lambdaa
        self.cost = cost
        self.Create_Model()
          
    def Create_Model(self):
        self.model = ConcreteModel()
        self.model.nodes   = Set(initialize = range(0,1))
        self.model.x       = Var(self.model.nodes, within=NonNegativeIntegers)
  
        def obj_rule(model):
            first_term  = self.model.x[0]**2*self.cost[0] + self.lambdaa[0]*self.model.x[0]*self.cost[1] + self.lambdaa[1]*self.model.x[0]*self.cost[2]
            return first_term 
        
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
     
    def solve(self, solver_name, display_solution_stream=False):
        instance = self.model
        opt = SolverFactory(solver_name)
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        return value(instance.x[0] )
    
################### SET THE SOLVER#############################################
solver_name =  "BONMIN"
#Number of VAriables 
number_of_variables = 6
#Initializing Lambda
lambda_ = np.array([1 , 1])
#Initial Problem cost
a_1 = np.array([-1,0.2,-1,0.2,-1,0.2])
a_2 = np.array([-5,1,-5,1,-5,1])
obj_cost = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
#Sub problem costs
cost_sp = [[0.5, -1, -5],[0.1, 0.2, 1], [0.5, -1, -5],[0.1, 0.2, 1],[0.5, -1, -5], [0.1, 0.2, 1]]   
#SLR initial paramters
alpha = 1
M = 100
r = 0.5
ItrNum = 1
####################SOLVING A RELAXED PROBLEM TO GET q_0######################## 
#Solving a relaxed problem to get x_0 and 1_0
gen_relaxed_problem=Central_Test(number_of_variables, a_1, a_2, obj_cost, NonNegativeReals)
display_solver_log = True
q_0, x_0 = gen_relaxed_problem.solve(solver_name, display_solver_log)
# Evaluating the x_0 solution in lagrangian function
Lagrang = sum(obj_cost*x_0 + lambda_[0]*a_1*x_0 + lambda_[1]*a_2*x_0) + 48*lambda_[0] + 250*lambda_[1]
g_x_0 = sum(a_1*x_0) + 48
g_x_1 = sum(a_2*x_0) + 250
#calculate 
c_0 = np.array([(q_0-Lagrang)/g_x_0**2, (q_0-Lagrang)/g_x_1**2 ])

lambda_ = lambda_ + c_0*np.array([g_x_0, g_x_1])
lambda_[lambda_<0] = 0
sub_sol = np.array([])
for k in range(0, ItrNum):
    for sub in range(0,6):
          sp = Sub_Problem(lambda_, cost_sp[sub])
          x_sp=sp.solve(solver_name, True)
          sub_sol = np.append(sub_sol, [x_sp])
          
    Lagrang = sum(obj_cost*sub_sol + lambda_[0]*a_1*sub_sol + lambda_[1]*a_2*sub_sol) + 48*lambda_[0] + 250*lambda_[1]
