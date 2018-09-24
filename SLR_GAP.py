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
    
    def __init__(self,nbM,nbJ):
        self.nbM = nbM
        self.nbJ = nbJ
        self.cost, self.b = self.gen_rand_problem()
        self.Create_Model()
    
    def gen_rand_problem(self):
        cost_ = np.empty([self.nbM,self.nbJ], dtype=float)
        cap = np.empty([self.nbM], dtype=float)
        for M in range(0, self.nbM):
            cap[M] = random.randint(100,300)
            for J in range(0, self.nbJ):
                cost_[M,J] = random.randint(0,100)       
        return cost_, cap
        
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
        opt = SolverFactory("Cplex" )
        opt.relax=1
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        for p in instance.nbM:
            for q in instance.nbJ:
                if instance.x[p,q].value==1:
                    print(p,q,instance.x[p,q].value)
        return instance 
            
        return results


class SLR_GAP:
    
    def __init__(self, lambdaa, node_list):
        self.lambdaa = lambdaa
        self.node_list = node_list
        self.Create_Model()
          
    def Create_Model(self):
        self.model = ConcreteModel()
        self.model.nodes   = Set(initialize = self.node_list)
        self.model.cost    = Param(self.model.nodes*self.model.nodes, initialize=random.randint(5,10))
        self.model.b       = Param(self.model.nodes, initialize=random.randint(0,10))
        self.model.x       = Var(self.model.nodes*self.model.nodes, within=Binary)
  
        def obj_rule(model):
            first_term  = sum(self.model.x[n]*(self.model.cost[n]+self.lambdaa[n[0]]) for n in self.model.nodes*self.model.nodes)
            second_term = sum(self.lambdaa[n[0]] for n in self.model.nodes*self.model.nodes) 
            return first_term - second_term
            
        self.model.obj     = Objective(rule=obj_rule,sense=minimize)
            
        def flow_balance_1(model,p):
            return sum(self.model.x[n,p] for n in self.model.nodes)<= self.model.b[p] 
                       
        self.model.flowbal_1 = Constraint(self.model.nodes, rule=flow_balance_1)
        self.model.flowbal_2 = Constraint(self.model.nodes, rule=flow_balance_1)
            
        
    def solve(self, display_solution_stream=False , solve_relaxation = True):
        instance = self.model
        instance.preprocess()
        instance.display()
        opt = SolverFactory("GLPK" )
        opt.relax = True
        results = opt.solve(instance, tee=display_solution_stream)
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')
        instance.solutions.store_to(results)
        for p in instance.nodes:
            for q in instance.nodes:
                if instance.x[p,q].value != 0:
                    print(p,q,instance.x[p,q].value)
        return instance 
    
    
lambdaa = [-10 for i in range(0,10)] 
node    = [i for i in range(0,10)]
a=Central_GAP(5,6)
#a=SLR_GAP(lambdaa, node)
b=a.solve(True, True)
