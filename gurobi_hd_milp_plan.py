import gurobipy as gp
from gurobipy import GRB

def initialize_variables(m: gp.Model, A, S, Aux, relus, A_type, S_type, Aux_type, horizon):
    
    VARINDEX = 0
    colnames = []
    GRBINF = GRB.INFINITY
    
    # Create vars for each action a, time step t
    x = {}
    for index, a in enumerate(A):
        for t in range(horizon):
            x[(a, t)] = m.addVar(name=str(VARINDEX), vtype=A_type[index], lb=(-1.0 * GRBINF), ub=GRBINF)
            colnames.append(x[(a, t)])
            VARINDEX += 1

    # Create vars for each state s, time step t
    y = {}
    for index, s in enumerate(S):
        for t in range(horizon + 1):
            y[(s, t)] = m.addVar(name=str(VARINDEX), vtype=S_type[index], lb=(-1.0 * GRBINF), ub=GRBINF)
            colnames.append(y[(s, t)])
            VARINDEX += 1

    # Create vars for each auxilary variable aux, time step t
    v = {}
    for index, aux in enumerate(Aux):
        for t in range(horizon + 1):
            v[(aux,t)] = m.addVar(name=str(VARINDEX), vtype=Aux_type[index], lb=(-1.0 * GRBINF), ub=GRBINF)
            colnames.append(v[(aux,t)])
            VARINDEX += 1

    # Create vars for each relu node z, time step t
    z = {}
    zPrime = {}
    for relu in relus:
        for t in range(horizon):
            z[(relu, t)] = m.addVar(name=str(VARINDEX), vtype="C", lb=(-1.0 * GRBINF), ub=GRBINF)
            colnames.append(z[(relu, t)])
            VARINDEX += 1
            zPrime[(relu,t)] = m.addVar(name=str(VARINDEX), vtype="B", lb=(-1.0 * GRBINF), ub=GRBINF)
            colnames.append(zPrime[(relu,t)])
            VARINDEX += 1
    
    return m, x, y, v, z, zPrime, colnames

def encode_global_constraints(m: gp.Model, constraints, A, S, Aux, x, y, v, horizon):
    
    for t in range(horizon):
        for constraint in constraints:
            variables = constraint[:-2]
            literals = []
            coefs = []
            
            for var in variables:
                
                coef = "1.0"
                
                if "*" in var:
                    coef, var = var.split("*")
                if var in A:
                    literals.append(x[(var, t)])
                    coefs.append(float(coef))
                elif var in S:
                    literals.append(y[(var, t)])
                    coefs.append(float(coef))
                else:
                    literals.append(v[(var, t)])
                    coefs.append(float(coef))
            
            RHS = float(constraint[len(constraint) - 1])
            
            if "<=" == constraint[len(constraint) - 2]:
                m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.LESS_EQUAL, rhs=RHS)
            elif ">=" == constraint[len(constraint) - 2]:
                m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.GREATER_EQUAL, rhs=RHS)
            else:
                m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.EQUAL, rhs=RHS)
                
    return m

def encode_initial_constraints(m: gp.Model, initial, y):
    
    for init in initial:
        variables = init[:-2]
        literals = []
        coefs = []
        
        for var in variables:
            
            coef = "1.0"
            
            if "*" in var:
                coef, var = var.split("*")
                
            literals.append(y[(var, 0)])
            coefs.append(float(coef))
            
        RHS = float(init[len(init) - 1])
        if "<=" == init[len(init) - 2]:
            m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.LESS_EQUAL, rhs=RHS)
        elif ">=" == init[len(init) - 2]:
            m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.GREATER_EQUAL, rhs=RHS)
        else:
            m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.EQUAL, rhs=RHS)
    
    return m

def encode_goal_constraints(m: gp.Model, goals, S, Aux, y, v, horizon):
    
    for goal in goals:
        variables = goal[:-2]
        literals = []
        coefs = []
        
        for var in variables:
            coef = "1.0"
            if "*" in var:
                coef, var = var.split("*")
            if var in S:
                literals.append(y[(var, horizon)])
                coefs.append(float(coef))
            else:
                literals.append(v[(var, horizon)])
                coefs.append(float(coef))
                
        RHS = float(goal[len(goal)-1])
 
        if "<=" == goal[len(goal) - 2]:
            m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.LESS_EQUAL, rhs=RHS)
        elif ">=" == goal[len(goal) - 2]:
            m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.GREATER_EQUAL, rhs=RHS)
        else:
            m.addLConstr(gp.LinExpr(coefs, literals), sense=GRB.EQUAL, rhs=RHS)
            

    return m

def encode_bigm_activation_constraints(m: gp.Model, relus, bias, inputNeurons, mappings, weights, A, S, x, y, z, zPrime, bigM, horizon):
    
    for t in range(horizon):
        for relu in relus:
            
            m.addLConstr(z[(relu, t)] >= 0)
            
            m.addLConstr(z[(relu, t)] - (1.0 * bigM * zPrime[(relu, t)]) <= 0)
            
            inputs = []
            coefs = []
            RHS = -1.0 * bias[relu]
            
            for inp in inputNeurons[relu]:
                if inp in mappings:
                    coefs.append(weights[(inp, relu)])
                    if mappings[inp] in A:
                        inputs.append(x[(mappings[inp], t)])
                    else:
                        inputs.append(y[(mappings[inp], t)])
                else:
                    coefs.append(weights[(inp, relu)])
                    inputs.append(z[(inp, t)])
        
            # Constraint: sum(coef * input) - z[(relu, t)] <= RHS
            row_expr = gp.LinExpr(coefs + [-1.0], inputs + [z[(relu, t)]])
            m.addLConstr(row_expr, sense=GRB.LESS_EQUAL, rhs=RHS)
            
            RHS += -1.0 * bigM
            
            # Constraint: sum(coef * input) - z[(relu, t)] - bigM * zPrime[(relu, t)] >= RHS - bigM
            row_expr = gp.LinExpr(coefs + [-1.0, -1.0 * bigM], inputs + [z[(relu, t)], zPrime[(relu, t)]])
            m.addLConstr(row_expr, sense=GRB.GREATER_EQUAL, rhs=RHS)
    
    return m

def encode_indicator_activation_constraints(m: gp.Model, relus, bias, inputNeurons, mappings, weights, A, S, x, y, z, zPrime: dict[tuple, gp.Var], bigM, horizon):
    
    for t in range(horizon):
        for relu in relus:
            
            m.addLConstr(z[(relu, t)] >= 0)
            
            inputs = []
            coefs = []
            RHS = -1.0 * bias[relu]
            
            for inp in inputNeurons[relu]:
                if inp in mappings:
                    coefs.append(weights[(inp, relu)])
                    if mappings[inp] in A:
                        inputs.append(x[(mappings[inp], t)])
                    else:
                        inputs.append(y[(mappings[inp], t)])
                else:
                    coefs.append(weights[(inp, relu)])
                    inputs.append(z[(inp, t)])
        
            # Constraint: sum(coef * input) - z[(relu, t)] <= RHS
            input_row_expr = gp.LinExpr(coefs, inputs)
            row_expr = gp.LinExpr(coefs + [-1.0], inputs + [z[(relu, t)]])
            m.addLConstr(row_expr, sense=GRB.LESS_EQUAL, rhs=RHS)
            
            # Input
            m.addConstr((zPrime[(relu, t)] == 0.0) >> (input_row_expr <= RHS))
            
            m.addConstr((zPrime[(relu, t)] == 1.0) >> (input_row_expr >= RHS))
            
            # Output
            m.addConstr((zPrime[(relu, t)] == 0.0) >> (z[(relu, t)] <= 0.0))
            
            m.addConstr((zPrime[(relu, t)] == 1.0) >> (row_expr >= RHS))

    
    return m

def encode_nextstate_constraints(m: gp.Model, outputs, bias, inputNeurons, mappings, weights, A, S, x, y, z, activationType, S_type, bigM, horizon):
    
    for t in range(1, horizon + 1):
        for output in outputs:
            
            inputs = []
            coefs = []
            RHS = -1.0 * bias[output]
            
            for inp in inputNeurons[output]:
                if inp in mappings:
                    coefs.append(weights[(inp, output)])
                    if mappings[inp] in A:
                        inputs.append(x[(mappings[(inp)], t - 1)])
                    else:
                        inputs.append(y[(mappings[(inp)], t - 1)])
                else:
                    coefs.append(weights[(inp, output)])
                    inputs.append(z[(inp, t - 1)])
        
            if activationType[output] == "linear" and S_type[S.index(mappings[output])] == "C":
                # Constraint: sum(coef * input) - y[(output, t)] = RHS
                row = gp.LinExpr(coefs + [-1.0], inputs + [y[(mappings[output], t)]])
                m.addLConstr(row, sense=GRB.EQUAL, rhs=RHS)
            
            elif activationType[(output)] == "linear" and S_type[S.index(mappings[(output)])] == "I":
                # Constraint: sum(coef * input) - y[(output, t)] <= RHS + 0.5
                row1 = gp.LinExpr(coefs + [-1.0], inputs + [y[(mappings[output], t)]])
                m.addLConstr(row1, sense=GRB.LESS_EQUAL, rhs=(RHS + 0.5))
                # Constraint: sum(coef * input) - y[(output, t)] >= RHS - 0.5
                row2 = gp.LinExpr(coefs + [-1.0], inputs + [y[(mappings[output], t)]])
                m.addLConstr(row2, sense=GRB.GREATER_EQUAL, rhs=(RHS - 0.5))
           
            elif activationType[(output)] == "step" and S_type[S.index(mappings[(output)])] == "B":
                # Constraint: sum(coef * input) - y[(output, t)] <= RHS
                row1 = gp.LinExpr(coefs + [-1.0 * bigM], inputs + [y[(mappings[output], t)]])
                m.addLConstr(row1, sense=GRB.LESS_EQUAL, rhs=RHS)
                # Constraint: sum(coef * input) - y[(output, t)] >= RHS - bigM
                row2 = gp.LinExpr(coefs + [-1.0 * bigM], inputs + [y[(mappings[output], t)]])
                m.addLConstr(row2, sense=GRB.GREATER_EQUAL, rhs=(RHS - bigM))
                
            else:
                print ("This activation function/state domain combination is currently not supported.")

    return m




def encode_reward(m: gp.Model, reward, colnames: list[gp.Var], A, S, Aux, x, y, v, horizon):
    
    objcoefs = [0.0] * len(colnames)
    # print(f"{colnames=}")
    m.update()
    colname_names = [var.VarName for var in colnames]
    
    for t in range(horizon):
        for var, weight in reward:
            if var in A:
                objcoefs[colname_names.index(x[(var, t)].varName)] = -1.0 * float(weight)
            elif var in S or var[:-1] in S:
                if var[len(var) - 1] == "'":
                    objcoefs[colname_names.index(y[(var[:-1], t + 1)].varName)] = -1.0 * float(weight)
                else:
                    objcoefs[colname_names.index(y[(var, t)].varName)] = -1.0 * float(weight)
            else:
                if var[len(var) - 1] == "'":
                    objcoefs[colname_names.index(v[(var[:-1], t + 1)].varName)] = -1.0 * float(weight)
                else:
                    objcoefs[colname_names.index(v[(var, t)].varName)] = -1.0 * float(weight)
    linobj = gp.LinExpr()
    for index, obj in enumerate(objcoefs):
        linobj += obj * colnames[index]
        
    m.setObjective(expr=linobj, sense=GRB.MINIMIZE)
    m.update()
    return m
