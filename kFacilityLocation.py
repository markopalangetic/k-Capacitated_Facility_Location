#%%
import numpy as np
import gurobipy as gb
import time


#%%
def distance (x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    res =  np.sqrt(np.sum((x_ext - y_ext)**2, axis=-1))
    return res


# %%
def embedd (distances):
    numPoints = distances.shape[0]
    const = 4
    size = int(np.log(numPoints))
    dim = const*size
    points = []
    for i in range(const):
        for j in range(size):
            treshold = 1/(2**j)
            rand = np.random.rand(numPoints)
            set = np.where(rand < treshold)[0]
            points.append(np.min(distances[:,set], -1))
    
    points = np.array(points).T

    repeat = dim*dim*numPoints
    distorsion = 1.05
    inds = np.random.choice(numPoints, (repeat, 2))
    for ind in inds:
        #inds = np.random.choice(1839, 2, replace=False)
        i = ind[0]
        j = ind[1]
        if i == j:
            continue
        ratio = distance(points[i], points[j])/distances[i][j]
        middle = (points[i] + points[j])/2
        if(ratio > distorsion):
            points[i] = points[i] + (1-distorsion/ratio)*(middle - points[i])
            points[j] = points[j] + (1-distorsion/ratio)*(middle - points[j])
        elif(ratio < 1/distorsion):
            points[i] = middle + 1/(distorsion*ratio)*(points[i] - middle)
            points[j] = middle + 1/(distorsion*ratio)*(points[j] - middle)
    return points

# %%
def initCenterSeed (points, numSD, capacities):
    initCenters = set()
    numA = points.shape[0]
    tmp_dists = np.ones(numA)
    while len(initCenters) < numSD:
        probs  = tmp_dists*capacities
        probs =  probs/np.sum(probs)
        cent = np.random.choice(numA, p = probs)
        if cent not in initCenters:
            initCenters.add(cent)
        
        tmp_dists = np.min(distance(points, points[list(initCenters)]), -1)
    return points[list(initCenters)]

#%%

def LinProgramStep (capacities, capBound, distances):
    numA = distances.shape[0]
    numSD = distances.shape[1]
    model = gb.Model('LinProgramStep')
    model.modelSense = gb.GRB.MINIMIZE
    model.Params.outputFlag = 0
    tranport  = []
    for i in range(numA):
        tranport.append([])
        for j in range(numSD):
            tranport[i].append(model.addVar(lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, obj=distances[i][j]))

    for i in range(numA):
        exp = gb.quicksum([tranport[i][j] for j in range(numSD)])
        model.addConstr(exp == capacities[i])
    
    for  j in range(numSD):
        exp = gb.quicksum([tranport[i][j] for i in range(numA)])
        model.addConstr(exp <= capBound)
    
    model.optimize()
    objective = model.ObjVal

    outputTransport = np.zeros((numA,  numSD))
    for i in range(numA):
        for j in range(numSD):
            outputTransport[i][j] = tranport[i][j].x
    
    return objective, outputTransport

# %%
def getCenters (points, initCenters, capacities, capBound):
    current_obj = 1e15
    centers = initCenters
    dists = distance(points, centers)**2
    while True:
        objective, outputTransport = LinProgramStep (capacities, capBound, dists)
        if np.abs(objective - current_obj) < 1e-4:
            break
        current_obj = objective
        weights = outputTransport/np.sum(outputTransport, axis=0)
        centers = weights.T @ points
        dists = distance(points, centers)**2

    centerIndices = np.argmin(dists, 0)
    return centerIndices

# %%
def LPEvaluation (capacities, capBound, distances, marketPrice, energyPrice, minJunk, 
                        minEnergy, operationPrice, junkPrice, buildingPrice, transportPrice,
                        loryCapacity, operationCost):

    numA = distances.shape[0]
    numSD = distances.shape[1]
    model = gb.Model('LPEvaluation')
    model.modelSense = gb.GRB.MAXIMIZE
    model.Params.outputFlag = 0

    
    aEnergy = model.addMVar (numA, vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY, obj=energyPrice)
    aJunk = model.addMVar (numA, vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY, obj=-junkPrice)

    if numSD > 0:
        sEnergy = model.addMVar (numSD, vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY, obj=energyPrice)
        dEnergy = model.addMVar (numSD, vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY, obj=energyPrice)

        sJunk = model.addMVar (numSD, vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY, obj=-junkPrice)
        dJunk = model.addMVar (numSD, vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY, obj=-junkPrice)

        sInput = model.addMVar (numSD, vtype = gb.GRB.CONTINUOUS, lb=0, ub=capBound, obj=-operationCost)
        dInput = model.addMVar (numSD, vtype = gb.GRB.CONTINUOUS, lb=0, ub=capBound, obj=-operationCost)

        tranportASD = model.addMVar((numA, numSD), vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY,
                                    obj = -distances*transportPrice/loryCapacity)


        dOutput = model.addMVar (numSD, vtype = gb.GRB.CONTINUOUS, lb=0, ub=gb.GRB.INFINITY, obj=marketPrice)


    model.addConstrs(aEnergy[i] <= capacities[i]  for i in range(numA))
    model.addConstrs(aEnergy[i] >= minEnergy*capacities[i]  for i in range(numA))

    model.addConstrs(aJunk[i] <= capacities[i]  for i in range(numA))
    model.addConstrs(aJunk[i] >= minJunk*capacities[i]  for i in range(numA))

    if numSD > 0:

        model.addConstrs(sEnergy[i] <= sInput[i]  for i in range(numSD))
        model.addConstrs(sEnergy[i] >= minEnergy*sInput[i]  for i in range(numSD))
        model.addConstrs(dEnergy[i] <= dInput[i]  for i in range(numSD))
        model.addConstrs(dEnergy[i] >= minEnergy*dInput[i]  for i in range(numSD))
        
        
        model.addConstrs(sJunk[i] <= sInput[i]  for i in range(numSD))
        model.addConstrs(sJunk[i] >= minJunk*sInput[i]  for i in range(numSD))
        model.addConstrs(dJunk[i] <= dInput[i]  for i in range(numSD))
        model.addConstrs(dJunk[i] >= minJunk*dInput[i]  for i in range(numSD))


        model.addConstrs(tranportASD[:,i] @ np.ones(numA) == sInput[i] for i in range(numSD))
        model.addConstrs(tranportASD[i] @ np.ones(numSD) <= capacities[i] - aEnergy[i] - aJunk[i] for i in range(numA))

        model.addConstrs(dInput[i] == sInput[i] - sEnergy[i] - sJunk[i] for i in range(numSD))
        model.addConstrs(dOutput[i] == dInput[i] - dEnergy[i] - dJunk[i] for i in range(numSD))

    model.optimize()
    return model.ObjVal - 2*numSD*buildingPrice

#%%
def facilityLocationHeuristics (capacities, capBound, distances, marketPrice, energyPrice, minJunk, 
                        minEnergy, operationPrice, junkPrice, buildingPrice, transportPrice,
                        loryCapacity, operationCost):
    start = time.time()
    points = embedd(np.sqrt(distances))
    numSD = 1
    best_obj = 0
    totalCapacity = np.sum(capacities)
    while True:
        tmp_best = 0
        for i in range(3):
            initCenters = initCenterSeed(points, numSD, capacities)
            if numSD*capBound >= (1 - minEnergy - minJunk)*totalCapacity:
                centers = getCenters(points, initCenters, (1 - minEnergy - minJunk)*capacities, capBound)
            else:
                centers = []
            obj = LPEvaluation(capacities, capBound, distances[:,centers], marketPrice, energyPrice, minJunk, 
                        minEnergy, operationPrice, junkPrice, buildingPrice, transportPrice,
                        loryCapacity, operationCost)
            if obj > tmp_best:
                tmp_best = obj
        if tmp_best < best_obj:
            break
        else:
            best_obj = tmp_best
            numSD = numSD + 1
    end = time.time()
    return best_obj, (end - start), numSD - 1

# %%
