# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:47:01 2018

@author: mshahabi
"""

"""
Solving a small problem with Surrogate Lagrangian Relaxation

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
import matplotlib.pyplot as plt


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
        instance.pprint
        opt = SolverFactory(solver_name)
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        return value(instance.x[0] )
    

      
################### SET THE SOLVER#############################################
solver_name =  "BONMIN"#Number of VAriables 
number_of_variables = 6
 #Initializing Lambda0
lambda_acum = {"x": [], "y": []}
lambda_ = np.array([0.1357254,  1.1019696])
lambda_acum["x"].append(lambda_[0])
lambda_acum["y"].append(lambda_[1])
#Initial Problem cost
a_1 = np.array([-1,0.2,-1,0.2,-1,0.2])
a_2 = np.array([-5,1,-5,1,-5,1])
obj_cost = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
#Sub problem costs
cost_sp = [[0.5, -1, -5], [0.1, 0.2, 1], [0.5, -1, -5], [0.1, 0.2, 1], [0.5, -1, -5], [0.1, 0.2, 1]]   
#SLR initial paramters
alpha = 0.9
M = 50
r = 0.8
ItrNum = 25
##################EXACT RESULTS################################################
gen_exact_problem = Central_Test(number_of_variables, a_1, a_2, obj_cost, NonNegativeIntegers)
display_solver_log = False
q_e, x_e = gen_exact_problem.solve(solver_name, display_solver_log)  
####################SOLVING CONTNIOUS RELAXATION PROBLEM TO GET q_0######################## 
#Solving a relaxed problem to get x_0 and 1_0
gen_relaxed_problem = Central_Test(number_of_variables, a_1, a_2, obj_cost, NonNegativeReals)
display_solver_log = False
q_0, x_r = gen_relaxed_problem.solve(solver_name, display_solver_log)  
####################SOLVING THE Lagrangian RELAXATION PROBLEM TO GET Lagrangian Value######################## 
sub_sol = np.empty([6],dtype=float)
Lagrang = 0
for sub in range(0,6):        
     sp = Sub_Problem(lambda_, cost_sp[sub])
     x_l=sp.solve(solver_name, False)
     Lagrang = obj_cost[sub]*x_l**2 + lambda_[0]*a_1[sub]*x_l + lambda_[1]*a_2[sub]*x_l + Lagrang
     sub_sol[sub] = x_l
# Evaluating the x_0 solution in lagrangian function
Lagrang = Lagrang + 48*lambda_[0] + 250*lambda_[1] 
g_x_0 = sum((a_1*sub_sol)) + 48
g_x_1 = sum((a_2*sub_sol)) + 250
g_x_old = np.array([g_x_0, g_x_1])
#calculate 
c_k_old = (q_0-Lagrang)/sum((g_x_old**2))

lambda_ = lambda_ + c_k_old*g_x_old
lambda_[lambda_<0] = 0
even_flag = 0
for k in range(1, ItrNum):
    if k%2 ==0: 
        even_flag = 1 
    else:even_flag = 0
    lambda_acum["x"].append(lambda_[0])
    lambda_acum["y"].append(lambda_[1]) 
    for sub in range(0,6):        
          if (even_flag == 1 and sub%2 ==1):
#              x_sp = test_solution(lambda_,cost_sp[sub])
              sp = Sub_Problem(lambda_, cost_sp[sub])
              x_sp=sp.solve(solver_name, False)
              sub_sol[sub] = x_sp
          elif(even_flag == 0 and sub%2 ==0):
#              x_sp = test_solution(lambda_,cost_sp[sub])
              sp = Sub_Problem(lambda_, cost_sp[sub])
              x_sp=sp.solve(solver_name, False)
              sub_sol[sub] = x_sp
    print(sub_sol)          
    g_x_0 = sum((a_1*sub_sol)) + 48
    g_x_1 = sum((a_2*sub_sol)) + 250
    g_x_new = np.array([g_x_0 , g_x_1])
    p = 1 - 1/(k**r)
    alpha = 1- 1/(M*k**p)
    c_k = alpha*c_k_old*(sum((g_x_old**2))/sum((g_x_new**2)))
    obj_value = (obj_cost*sub_sol**2)
    obj_value = sum(obj_value)
    if (c_k*g_x_new.all()) <= 0.0000000000001 : break
    lambda_ = lambda_ + c_k*g_x_new
    lambda_[lambda_<0] = 0
    g_x_old = g_x_new
    c_k_old = c_k
    
# test functin to test the value of the optimitionproblem
def test_solution(lambdaa,cost_):
    b = lambdaa[0]*cost_[1] + lambdaa[1]*cost_[2]
    x_1 =  -  b/(2*cost_[0])
    if x_1<0:x_1=0 
    return int(x_1)   


def plot_fun(lambda_):
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.plot(lambda_["x"], lambda_["y"],'-o')
      ax.set_xlabel('Lambda 1')
      ax.set_ylabel('Lambda 0')
      plt.show()    
plot_fun(lambda_acum)      