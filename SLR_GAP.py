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

class Central_GAP:
    
    def __init__(self):
        self.Create_Model()
   
    def Create_Model(self):
        self.model = ConcreteModel()
        self.model.nodes   = Set(initialize=range(0,10))
        self.model.cost    = Param(self.model.nodes*self.model.nodes, initialize=random.randint(0,10))
        self.model.b       = Param(self.model.nodes, initialize=random.randint(0,10))
        self.model.x       = Var(self.model.nodes*self.model.nodes, within=Binary)
        self.model.pprint()    
        def obj_rule(model):
            first_term  = sum(self.model.x[n]*self.model.cost[n] for n in self.model.nodes*self.model.nodes)          
            return first_term
            
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
            
        def flow_balance_1(model,p):
            return sum(self.model.x[n,p] for n in self.model.nodes)<= b[p] 
        
        def flow_balance_2(model,n):          
            return sum(self.model.x[n,p] for p in self.model.nodes)==1 
        
                
        self.model.flowbal_1 = Constraint(self.model.nodes, rule=flow_balance_1)
        self.model.flowbal_2 = Constraint(self.model.nodes, rule=flow_balance_1)
            
        
    def solve(self):
        instance = self.model
        instance.display()
        opt = SolverFactory("IPOPT")
        results = opt.solve(instance)
        test_flow = 0
            
        return results


class SLR_GAP:
    
    def __init__(self, lambdaa):
        self.Create_Model()
        self.lambdaa = lambdaa
   
    def Create_Model(self):
        self.model = ConcreteModel()
        self.model.nodes   = Set(initialize=range(0,10))
        self.model.cost    = Param(self.model.nodes*self.model.nodes, initialize=random.randint(0,10))
        self.model.b       = Param(self.model.nodes, initialize=random.randint(0,10))
        self.model.x       = Var(self.model.nodes*self.model.nodes, within=Binary)
  
        def obj_rule(model):
            first_term  = sum(self.model.x[n]*(self.model.cost[n]+self.model.lambdaa[n]) for n in self.model.nodes*self.model.nodes)          
            return first_term
            
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
            
        def flow_balance_1(model,p):
            return sum(self.model.x[n,p] for n in self.model.nodes)<= self.model.b[p] 
                       
        self.model.flowbal_1 = Constraint(self.model.nodes, rule=flow_balance_1)
        self.model.flowbal_2 = Constraint(self.model.nodes, rule=flow_balance_1)
            
        
    def solve(self):
        instance = self.model
        instance.display()
        opt = SolverFactory("IPOPT")
        results = opt.solve(instance)
        test_flow = 0
            
        return results    
a=SLR_GAP()
b=a.solve()