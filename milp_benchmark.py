import cplex

import os

import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import gurobipy as gp
from gurobipy import GRB

import hd_milp_plan as c_plan
import gurobi_hd_milp_plan as g_plan


load_dotenv()

options = {
    "WLSACCESSID": os.environ.get("WLSACCESSID"),
    "WLSSECRET": os.environ.get("WLSSECRET"),
    "LICENSEID": int(os.environ.get("LICENSEID")),
}
ENV = gp.Env(params=options)




def encode_cplex_hd_milp_plan(domain, instance, horizon, sparsification, bound):
    
    bigM = 1000000.0
    
    inputNeurons, weights, bias, activationType = c_plan.readDNN(f"./dnn/dnn_{domain}_{instance}.txt")
    initial = c_plan.readInitial(f"./translation/initial_{domain}_{instance}.txt")
    goal = c_plan.readGoal(f"./translation/goal_{domain}_{instance}.txt")
    constraints = c_plan.readConstraints(f"./translation/constraints_{domain}_{instance}.txt")
    A, S, Aux, A_type, S_type, Aux_type = c_plan.readVariables(f"./translation/pvariables_{domain}_{instance}.txt")
    mappings = c_plan.readMappings(f"./translation/mappings_{domain}_{instance}.txt")
    
    relus = [relu for relu in inputNeurons.keys() if activationType[(relu)] == "relu"]
    outputs = [output for output in inputNeurons.keys() if activationType[(output)] == "linear" or activationType[(output)] == "step"]
    
    transitions = []
    if len(outputs) < len(S):
        transitions = c_plan.readTransitions(f"./translation/transitions_{domain}_{instance}.txt")
    
    if sparsification > 0.0:
        weights, bias = c_plan.sparsifyDNN(sparsification, weights, bias, inputNeurons, mappings, relus, outputs)

    reward = c_plan.readReward(f"./translation/reward_{domain}_{instance}.txt")
    
    # CPLEX
    c = cplex.Cplex()

    # Set number of threads
    c.parameters.threads.set(1)

    # Initialize variables
    c, x, y, v, z, zPrime, vartypes, colnames = c_plan.initialize_variables(c, A, S, Aux, relus, A_type, S_type, Aux_type, horizon)

    # Set global constraints
    c = c_plan.encode_global_constraints(c, constraints, A, S, Aux, x, y, v, horizon)

    # Set initial state
    c = c_plan.encode_initial_constraints(c, initial, y)

    # Set goal state
    c = c_plan.encode_goal_constraints(c, goal, S, Aux, y, v, horizon)

    # Set node activations
    c = c_plan.encode_activation_constraints(c, relus, bias, inputNeurons, mappings, weights, A, S, x, y, z, zPrime, bigM, horizon)

    # Predict the next state using DNNs
    c = c_plan.encode_nextstate_constraints(c, outputs, bias, inputNeurons, mappings, weights, A, S, x, y, z, activationType, S_type, bigM, horizon)

    if bound == "True":
        # Set strengthened activation constraints
        c = c_plan.encode_strengthened_activation_constraints(c, A, S, relus, bias, inputNeurons, mappings, weights, colnames, x, y, z, zPrime, horizon)

    if len(outputs) < len(S):
        # Set known transition function
        c = c_plan.encode_known_transitions(c, transitions, A, S, Aux, x, y, v, horizon)

    # Reward function
    c = c_plan.encode_reward(c, reward, colnames, A, S, Aux, x, y, v, horizon)

    # Set time limit
    #c.parameters.timelimit.set(3600.0)
    
    # Set optimality tolerance
    #c.parameters.mip.tolerances.mipgap.set(0.2)
    
    c.solve()

    #c.write("hd_milp_plan.lp")

    solution = c.solution
    
    print("")

    if solution.get_status() == solution.status.MIP_infeasible:
        print("No plans w.r.t. the given DNN exists.")
    elif solution.get_status() == solution.status.MIP_optimal:
        print("An optimal plan w.r.t. the given DNN is found:")
        
        solX = solution.get_values()
        for s in S:
           print("%s at time %d by: %f " % (s,0,solX[y[(s,0)]]))
        for t in range(horizon):
            for a in A:
                print("%s at time %d by: %f " % (a,t,solX[x[(a,t)]]))
            for s in S:
               print("%s at time %d by: %f " % (s,t+1,solX[y[(s,t+1)]]))
    elif solution.get_status() == solution.status.MIP_feasible or solution.get_status() == solution.status.MIP_abort_feasible or solution.get_status() == solution.status.MIP_time_limit_feasible or solution.get_status() == solution.status.MIP_dettime_limit_feasible or solution.get_status() == solution.status.optimal_tolerance:
        print("A plan w.r.t. the given DNN is found:")
        
        solX = solution.get_values()
        for s in S:
           print("%s at time %d by: %f " % (s,0,solX[y[(s,0)]]))
        for t in range(horizon):
            for a in A:
                print("%s at time %d by: %f " % (a,t,solX[x[(a,t)]]))
            for s in S:
               print("%s at time %d by: %f " % (s,t+1,solX[y[(s,t+1)]]))
    elif solution.get_status() == solution.status.MIP_abort_infeasible:
        print("Planning is interrupted by the user.")
    elif solution.get_status() == solution.status.MIP_time_limit_infeasible:
        print("Planning is terminated by the time limit without a plan.")
    else:
        print("Planning is interrupted. See the status message: %d" % solution.get_status())

    print("")

    return


def encode_gurobi_hd_milp_plan(domain, instance, horizon, sparsification, bound):
    
    bigM = 1000000.0
    
    inputNeurons, weights, bias, activationType = c_plan.readDNN(f"./dnn/dnn_{domain}_{instance}.txt")
    initial = c_plan.readInitial(f"./translation/initial_{domain}_{instance}.txt")
    goal = c_plan.readGoal(f"./translation/goal_{domain}_{instance}.txt")
    constraints = c_plan.readConstraints(f"./translation/constraints_{domain}_{instance}.txt")
    A, S, Aux, A_type, S_type, Aux_type = c_plan.readVariables(f"./translation/pvariables_{domain}_{instance}.txt")
    mappings = c_plan.readMappings(f"./translation/mappings_{domain}_{instance}.txt")
    
    
    relus = [relu for relu in inputNeurons.keys() if activationType[(relu)] == "relu"]
    outputs = [output for output in inputNeurons.keys() if activationType[(output)] == "linear" or activationType[(output)] == "step"]
    
    transitions = []
    if len(outputs) < len(S):
        transitions = c_plan.readTransitions(f"./translation/transitions_{domain}_{instance}.txt")
    
    if sparsification > 0.0:
        weights, bias = c_plan.sparsifyDNN(sparsification, weights, bias, inputNeurons, mappings, relus, outputs)

    reward = c_plan.readReward(f"./translation/reward_{domain}_{instance}.txt")
    
    # try:
    m = gp.Model(env=ENV)
    
    # Set the number of threads to 1
    m.setParam('Threads', 1)
    
    # Initialize variables
    m, x, y, v, z, zPrime, colnames = g_plan.initialize_variables(m, A, S, Aux, relus, A_type, S_type, Aux_type, horizon)
    
    
    # Set global constraints
    m = g_plan.encode_global_constraints(m, constraints, A, S, Aux, x, y, v, horizon)
    
    
    # Set initial state
    m = g_plan.encode_initial_constraints(m, initial, y)
    
    
    # Set goal state
    m = g_plan.encode_goal_constraints(m, goal, S, Aux, y, v, horizon)
    
    
    # Set node activations
    # m = g_plan.encode_bigm_activation_constraints(m, relus, bias, inputNeurons, mappings, weights, A, S, x, y, z, zPrime, bigM, horizon)
    m = g_plan.encode_indicator_activation_constraints(m, relus, bias, inputNeurons, mappings, weights, A, S, x, y, z, zPrime, bigM, horizon)
    
    
    # Predict the next state using DNNs
    m = g_plan.encode_nextstate_constraints(m, outputs, bias, inputNeurons, mappings, weights, A, S, x, y, z, activationType, S_type, bigM, horizon)
    
    if bound == "True":
        # Set strengthened activation constraints
        c = c_plan.encode_strengthened_activation_constraints(c, A, S, relus, bias, inputNeurons, mappings, weights, colnames, x, y, z, zPrime, horizon)

    if len(outputs) < len(S):
        # Set known transition function
        c = c_plan.encode_known_transitions(c, transitions, A, S, Aux, x, y, v, horizon)
    # Reward function
    m = g_plan.encode_reward(m, reward, colnames, A, S, Aux, x, y, v, horizon)

    # Set time limit
    #c.parameters.timelimit.set(3600.0)
    
    # Set optimality tolerance
    #c.parameters.mip.tolerances.mipgap.set(0.2)
    
    m.optimize()
    
    data = {
        "Reward": m.ObjVal,
        "Time": m.Runtime,
        "Dual": m.ObjBoundC
    }
    
    
    # solX = [var.Xn for var in m.getVars()]
    # solX = [var.VarName for var in m.getVars()]
    
    # for t in range(horizon):
    #     for a in A:
    #         print(f"{x[(a,t)].VarName=}")
    #         print("%s at time %d by: %f " % (a,t,solX[int(x[(a,t)].VarName)]))
            
            
    print("")

    # if m.Status == GRB.Status.INFEASIBLE:
    #     print("No plans w.r.t. the given DNN exists.")
    # elif m.Status == GRB.Status.OPTIMAL:
    #     print("An optimal plan w.r.t. the given DNN is found:")
        
    #     solX = [var.Xn for var in m.getVars()]
        
    #     for s in S:
    #        print("%s at time %d by: %f " % (s, 0, solX[ int( y[(s,0)].VarName ) ]))
    #     for t in range(horizon):
    #         for a in A:
    #             print("%s at time %d by: %f " % (a, t, solX[ int( x[(a, t)].VarName ) ]))
    #         for s in S:
    #            print("%s at time %d by: %f " % (s,t+1,solX[ int( y[(s,t+1)].VarName ) ]))
    
    
    
    # elif solution.get_status() == solution.status.MIP_feasible or solution.get_status() == solution.status.MIP_abort_feasible or solution.get_status() == solution.status.MIP_time_limit_feasible or solution.get_status() == solution.status.MIP_dettime_limit_feasible or solution.get_status() == solution.status.optimal_tolerance:
    #     print("A plan w.r.t. the given DNN is found:")
        
    #     solX = solution.get_values()
    #     #for s in S:
    #     #    print("%s at time %d by: %f " % (s,0,solX[y[(s,0)]]))
    #     for t in range(horizon):
    #         for a in A:
    #             print("%s at time %d by: %f " % (a,t,solX[x[(a,t)]]))
    #         #for s in S:
    #         #    print("%s at time %d by: %f " % (s,t+1,solX[y[(s,t+1)]]))
    # elif solution.get_status() == solution.status.MIP_abort_infeasible:
    #     print("Planning is interrupted by the user.")
    # elif solution.get_status() == solution.status.MIP_time_limit_infeasible:
    #     print("Planning is terminated by the time limit without a plan.")
    # else:
    #     print("Planning is interrupted. See the status message: %d" % solution.get_status())

    print("")
    
    # except gp.GurobiError as e: 
    #     print('Error code ' + str(e.errno) + ': ' + str(e))
    # except AttributeError:
    #     print('Encountered an attribute error ')
    
    return



if __name__ == "__main__":
    
    myargs = c_plan.get_args()
    
    setDomain = False
    setInstance = False
    setHorizon = False
    setSparsification = False
    setBounds = False
    
    sparsification = "0.0"
    
    for arg in myargs:
        if arg == "-d":
            domain = myargs[(arg)]
            setDomain = True
        elif arg == "-i":
            instance = myargs[(arg)]
            setInstance = True
        elif arg == "-h":
            horizon = myargs[(arg)]
            setHorizon = True
        elif arg == "-s":
            sparsification = myargs[(arg)]
            setSparsification = True
        elif arg == "-b":
            bound = myargs[(arg)]
            setBounds = True

    if setDomain and setInstance and setHorizon and setBounds:
        # encode_cplex_hd_milp_plan(domain, instance, int(horizon), float(sparsification), bound)
        encode_gurobi_hd_milp_plan(domain, instance, int(horizon), float(sparsification), bound)
    elif not setDomain:
        print ('Domain is not provided.')
    elif not setInstance:
        print ('Instance is not provided.')
    elif not setHorizon:
        print ('Horizon is not provided.')
    else:
        print ('Bounding decision is not provided.')
