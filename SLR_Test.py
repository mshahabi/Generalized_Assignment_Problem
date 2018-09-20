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

class Central_Test:
    
    def __init__(self, number_of_x, a1, a2, obj_cost, Variable_Type):
        self.a1 = a1
        self.a2 = a2
        self.cost = obj_cost
        self.number_of_x = number_of_x
        self.Create_Model(Variable_Type)
       
   
    def Create_Model(self, Variable_Type):
        self.model = ConcreteModel()
        self.model.nodes   = Set(initialize=range(0, self.number_of_x))
        self.model.x       = Var(self.model.nodes, within = Variable_Type)   
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
        for p in instance.nodes:
                if instance.x[p].value!=0:
                    print(p,instance.x[p].value)
        return instance 
            
        return results


class Sub_Problem:
    
    def __init__(self, lambdaa):
        self.lambdaa = lambdaa
        self.Create_Model()
          
    def Create_Model(self):
        self.model = ConcreteModel()
        self.model.nodes   = Set(initialize = range(0,1))
        self.model.x       = Var(self.model.nodes, within=NonNegativeIntegers)
  
        def obj_rule(model):
            first_term  = sum(self.model.x[n] for 0 in range(0,3))
            return first_term - second_term
        
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
     
    def solve(self, solver_name, display_solution_stream=False):
        instance = self.model
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        if instance.x[0].value != 0:
              print(instance.x[0].value)
        return instance 
    
    
lambdaa = [-10 for i in range(0,10)] 
node    = [i for i in range(0,10)]
a1 = [-1,0.2,-1,0.2,-1,0.2]
a2 = [-5,1,-5,1,-5,1]
obj_cost = [0.5,0.1,0.5,0.1,0.5,0.1]
a=Central_Test(6, a1, a2, obj_cost,NonNegativeIntegers)
#a=SLR_GAP(lambdaa, n ode)
solver_name =  "BONMIN"
aa = Sub_Problem()
b=aa.solve(solver_name, True)
