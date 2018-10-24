# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:43:36 2018

@author: mshahabi
"""
config = {}
config["solver_name"] =  "cplex"
config["display_solution_stream"] = False
config["genrate_new_problem"] = True
config["relax_solution"] = False
config["alpha"] = 0.06
config["M"] = 40
config["r"] = 0.5
config["ItrNum"] = 120
config["num_of_machines"] = 40
config["num_of_jobs"] = 4000
config["num_of_sub_problems"] = 10