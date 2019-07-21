import numpy as np
import scipy as sp
import scipy.sparse

###
#Convenience
###

def corrcov(Cov):
    sd = np.sqrt(np.diag(Cov))
    invDiagSd = np.diag(1 / sd)
    Corr = invDiagSd @ Cov @ invDiagSd
    return Corr, sd

###
#Mappings
###

def generateMappings(indID, obsID, nInd, nObs, nRow):
    #Map observations to individuals
    obsIndID = np.unique(np.stack((indID, obsID), axis = 1), axis = 0)[:,0]
    sparseRow_map_obs_to_ind = np.arange(nObs)
    sparseCol_map_obs_to_ind = np.zeros((nObs,), dtype = 'int64')
    for i in np.arange(1, nObs):
        if obsIndID[i] == obsIndID[i - 1]:
            sparseCol_map_obs_to_ind[i] = sparseCol_map_obs_to_ind[i - 1]
        else:
            sparseCol_map_obs_to_ind[i] = sparseCol_map_obs_to_ind[i - 1] + 1
    map_obs_to_ind = scipy.sparse.csr_matrix((np.ones((nObs,), dtype = 'int64'), (sparseRow_map_obs_to_ind, sparseCol_map_obs_to_ind)), shape=(nObs, nInd))
    
    #Map alternatives to observations
    sparseRow_map_avail_to_obs = np.arange(nRow)
    sparseCol_map_avail_to_obs = np.zeros((nRow,), dtype = 'int64')
    for i in np.arange(1, nRow):
        if obsID[i] == obsID[i - 1]:
            sparseCol_map_avail_to_obs[i] = sparseCol_map_avail_to_obs[i - 1]
        else:
            sparseCol_map_avail_to_obs[i] = sparseCol_map_avail_to_obs[i - 1] + 1  
    map_avail_to_obs = scipy.sparse.csr_matrix((np.ones((nRow,), dtype = 'int64'), (sparseRow_map_avail_to_obs, sparseCol_map_avail_to_obs)), shape=(nRow, nObs))
    
    return map_obs_to_ind, map_avail_to_obs

###
#Prepare data
###
    
def prepareData(xList, indID, obsID, chosen, full = False):
    nInd = np.unique(indID).shape[0]
    nObs = np.unique(obsID).shape[0]
    nRowFull = indID.shape[0]
    
    map_obs_to_ind, map_avail_to_obs = generateMappings(indID, obsID, nInd, nObs, nRowFull)
    rowsPerObs = np.ndarray.flatten(np.ones((1, nRowFull), dtype = 'int64') @ map_avail_to_obs)
    
    chosenIdx = (np.arange(nRowFull) + 1) * chosen
    chosenIdx = chosenIdx[chosenIdx > 0] - 1
    nonChosenIdx = (np.arange(nRowFull) + 1) * (chosen == 0)
    nonChosenIdx = nonChosenIdx[nonChosenIdx > 0] - 1
    
    for j in range(len(xList)):
        x_tmp = xList[j]
        if x_tmp.shape[1] > 0:
            b = 0
            for i in np.arange(nObs):
                a = b; b += rowsPerObs[i]; idx = slice(a, b);
                x_tmp[idx, :] = x_tmp[idx, :] - x_tmp[chosenIdx[i], :] 
            if full == False: x_tmp = np.array(x_tmp[nonChosenIdx, :])
            xList[j] = x_tmp
    
    if full == False:
        map_avail_to_obs = scipy.sparse.csr_matrix(map_avail_to_obs[nonChosenIdx, :])
        nRow = nonChosenIdx.shape[0]
    else:
        nRow = nRowFull
    rowsPerInd = np.ndarray.flatten(np.ones((1, nRow), dtype = 'int64') @ map_avail_to_obs @ map_obs_to_ind)
    return (xList,
            nInd, nObs, nRow,
            chosenIdx, nonChosenIdx,
            rowsPerInd, rowsPerObs,
            map_obs_to_ind, map_avail_to_obs)
    
###
#Probabilities
###
    
def mvnlpdf(x, mu, Sigma):
    xS = (x - mu).T
    f = -0.5 * (xS * np.linalg.solve(Sigma, xS)).sum(axis = 0)
    return f

def transFix(paramFix, xFix_trans):
    paramFix_trans = np.array(paramFix)
    idx = xFix_trans == 1
    paramFix_trans[idx] = np.exp(paramFix[idx])
    return paramFix_trans
    
def transRnd(paramRnd, xRnd_trans):
    paramRnd_trans = np.array(paramRnd)
    
    idx = xRnd_trans == 1
    if np.sum(idx) > 0:
        paramRnd_trans[:, idx] = np.exp(paramRnd[:, idx])
        
    idx = xRnd_trans == 2
    if np.sum(idx) > 0:
        eP = np.exp(paramRnd[:, idx])
        paramRnd_trans[:, idx] = eP / (1 + eP)       
    return paramRnd_trans
    
def probMxl(
        paramFix, paramRnd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs):
    vFix = 0; vRnd = 0;
    if nFix > 0:
        if xFix_transBool: paramFix = transFix(paramFix, xFix_trans)
        vFix = xFix @ paramFix    
    if nRnd > 0:
        if xRnd_transBool: paramRnd = transRnd(paramRnd, xRnd_trans)
        paramRndPerRow = np.repeat(paramRnd, rowsPerInd, axis = 0)
        vRnd = np.sum(xRnd * paramRndPerRow, axis = 1)
            
    v = vFix + vRnd
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    nev = map_avail_to_obs.T @ ev + 1
    pChosen = 1 / nev
    lPChosen = np.log(pChosen)
    lPInd = map_obs_to_ind.T @ lPChosen
    return lPInd

###
#Prediction
###
    
def pPredMxl(vFix, vRnd, sim_map_avail_to_obs, D, chosenIdx, nonChosenIdx):
    v = vFix + vRnd
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    nev = sim_map_avail_to_obs.T @ ev + 1
    nnev = sim_map_avail_to_obs @ nev
    pChosen = 1 / nev
    pNonChosen = ev / nnev
    pPredChosen = pChosen.reshape((D, -1)).mean(axis = 0)
    pPredNonChosen = pNonChosen.reshape((D, -1)).mean(axis = 0)
    pPred = np.zeros((chosenIdx.shape[0] + nonChosenIdx.shape[0]))
    pPred[chosenIdx] = pPredChosen
    pPred[nonChosenIdx] = pPredNonChosen
    return pPred