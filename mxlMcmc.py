#from joblib import Parallel, delayed
import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.stats import invwishart
import scipy.sparse
from math import floor
import h5py

from mxl import corrcov, prepareData, mvnlpdf, probMxl, pPredMxl
                
###
#MCMC
###
    
def next_paramFix(
        paramFix, paramRnd,
        lPInd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
        rhoF):
    paramFix_star = paramFix + np.sqrt(rhoF) * np.random.randn(nFix,)
    lPInd_star = probMxl(
        paramFix_star, paramRnd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)
    r = np.exp(np.sum(lPInd_star - lPInd, axis = 0))
    if np.random.rand() <= r:
        paramFix = np.array(paramFix_star)
        lPInd = np.array(lPInd_star)
    return paramFix, lPInd
    
def next_zeta(paramRnd, Omega, nRnd, nInd):
    zeta = paramRnd.mean(axis = 0) + np.linalg.cholesky(Omega) @ np.random.randn(nRnd,) / np.sqrt(nInd)
    return zeta

def next_Omega(paramRnd, zeta, nu, iwDiagA, diagCov, nRnd, nInd):
    betaS = paramRnd - zeta
    Omega = np.array(invwishart.rvs(nu + nInd + nRnd - 1, 2 * nu * np.diag(iwDiagA) + betaS.T @ betaS)).reshape((nRnd, nRnd))
    if diagCov: Omega = np.diag(np.diag(Omega))
    return Omega

def next_iwDiagA(Omega, nu, invASq, nRnd):
    iwDiagA = np.random.gamma((nu + nRnd) / 2, 1 / (invASq + nu * np.diag(np.linalg.inv(Omega))))
    return iwDiagA

def next_paramRnd(
        paramFix, paramRnd, zeta, Omega,
        lPInd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
        rho):
    lPhi = mvnlpdf(paramRnd, zeta, Omega)
    paramRnd_star = paramRnd + np.sqrt(rho) * (np.linalg.cholesky(Omega) @ np.random.randn(nRnd, nInd)).T    
    lPInd_star = probMxl(
        paramFix, paramRnd_star,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)
    lPhi_star = mvnlpdf(paramRnd_star, zeta, Omega)

    r = np.exp(lPInd_star + lPhi_star - lPInd - lPhi)
    idxAccept = np.random.rand(nInd,) <= r

    paramRnd[idxAccept, :] = np.array(paramRnd_star[idxAccept, :])
    lPInd[idxAccept] = np.array(lPInd_star[idxAccept])

    acceptRate = np.mean(idxAccept)
    rho = rho - 0.001 * (acceptRate < 0.3) + 0.001 * (acceptRate > 0.3)
    return paramRnd, lPInd, rho

def mcmcChain(
        chainID, seed,
        mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
        rhoF, rho,
        modelName,
        paramFix, zeta, Omega, invASq, nu, diagCov,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd, 
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Precomputations
    ###
    
    if nRnd > 0:
        paramRnd = zeta + (np.linalg.cholesky(Omega) @ np.random.randn(nRnd, nInd)).T
        iwDiagA = np.random.gamma(1 / 2, 1 / invASq)
    else:
        paramRnd = np.zeros((0,0))
        iwDiagA = np.zeros((0,0))
    
    lPInd = probMxl(
            paramFix, paramRnd,
            xFix, xFix_transBool, xFix_trans, nFix, 
            xRnd, xRnd_transBool, xRnd_trans, nRnd,
            nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)   
    
    ###
    #Storage
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    if os.path.exists(fileName):
        os.remove(fileName) 
    file = h5py.File(fileName, "a")
    
    if nFix > 0:
        paramFix_store = file.create_dataset('paramFix_store', (mcmc_iterSampleThin, nFix))
        
        paramFix_store_tmp = np.zeros((mcmc_iterMemThin, nFix))
        
    if nRnd > 0:
        paramRnd_store = file.create_dataset('paramRnd_store', (mcmc_iterSampleThin, nInd, nRnd))
        zeta_store = file.create_dataset('zeta_store', (mcmc_iterSampleThin, nRnd))
        Omega_store = file.create_dataset('Omega_store', (mcmc_iterSampleThin, nRnd, nRnd))
        Corr_store = file.create_dataset('Corr_store', (mcmc_iterSampleThin, nRnd, nRnd))
        sd_store = file.create_dataset('sd_store', (mcmc_iterSampleThin, nRnd))
        
        paramRnd_store_tmp = np.zeros((mcmc_iterMemThin, nInd, nRnd))
        zeta_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
        Omega_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
        Corr_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
        sd_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
    
    ###
    #Sample
    ###
    
    j = -1
    ll = 0
    sampleState = 'burn in'
    for i in np.arange(mcmc_iter):
        if nFix > 0:
            paramFix, lPInd = next_paramFix(
                    paramFix, paramRnd,
                    lPInd,
                    xFix, xFix_transBool, xFix_trans, nFix, 
                    xRnd, xRnd_transBool, xRnd_trans, nRnd,
                    nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
                    rhoF)
            
        if nRnd > 0:
            zeta = next_zeta(paramRnd, Omega, nRnd, nInd)
            Omega = next_Omega(paramRnd, zeta, nu, iwDiagA, diagCov, nRnd, nInd)
            iwDiagA = next_iwDiagA(Omega, nu, invASq, nRnd)
            paramRnd, lPInd, rho = next_paramRnd(
                    paramFix, paramRnd, zeta, Omega,
                    lPInd,
                    xFix, xFix_transBool, xFix_trans, nFix, 
                    xRnd, xRnd_transBool, xRnd_trans, nRnd,
                    nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
                    rho)
        
        if ((i + 1) % mcmc_disp) == 0:
            if (i + 1) > mcmc_iterBurn:
                sampleState = 'sampling'
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (' + sampleState + ')')
            sys.stdout.flush()
            
        if (i + 1) > mcmc_iterBurn:   
            if ((i + 1) % mcmc_thin) == 0:
                j+=1
            
                if nFix > 0:
                    paramFix_store_tmp[j,:] = paramFix
            
                if nRnd > 0:
                    paramRnd_store_tmp[j,:,:] = paramRnd
                    zeta_store_tmp[j,:] = zeta
                    Omega_store_tmp[j,:,:] = Omega
                    Corr_store_tmp[j,:,:], sd_store_tmp[j,:,] = corrcov(Omega)
                    
            if (j + 1) == mcmc_iterMemThin:
                l = ll; ll += mcmc_iterMemThin; sl = slice(l, ll)
                
                print('Storing chain ' + str(chainID + 1))
                sys.stdout.flush()
                
                if nFix > 0:
                    paramFix_store[sl,:] = paramFix_store_tmp
                    
                if nRnd > 0:
                    paramRnd_store[sl,:,:] = paramRnd_store_tmp
                    zeta_store[sl,:] = zeta_store_tmp
                    Omega_store[sl,:,:] = Omega_store_tmp
                    Corr_store[sl,:,:] = Corr_store_tmp
                    sd_store[sl,:,] = sd_store_tmp
                
                j = -1 

###
#Posterior analysis
###  

def postAna(paramName, nParam, nParam2, mcmc_nChain, mcmc_iterSampleThin, modelName):
    colHeaders = ['mean', 'std. dev.', '2.5%', '97.5%', 'Rhat']
    q = np.array([0.025, 0.975])
    nSplit = 2
    
    postDraws = np.zeros((mcmc_nChain, mcmc_iterSampleThin, nParam, nParam2))
    for c in range(mcmc_nChain):
        file = h5py.File(modelName + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
        postDraws[c,:,:,:] = np.array(file[paramName + '_store']).reshape((mcmc_iterSampleThin, nParam, nParam2))
        
    tabPostAna = np.zeros((nParam * nParam2, len(colHeaders)))
    postMean = np.mean(postDraws, axis = (0,1))
    tabPostAna[:, 0] = np.array(postMean).reshape((nParam * nParam2,))
    tabPostAna[:, 1] = np.array(np.std(postDraws, axis = (0,1))).reshape((nParam * nParam2,))
    tabPostAna[:, 2] = np.array(np.quantile(postDraws, q[0], axis = (0,1))).reshape((nParam * nParam2,))
    tabPostAna[:, 3] = np.array(np.quantile(postDraws, q[1], axis = (0,1))).reshape((nParam * nParam2,))
    
    m = floor(mcmc_nChain * nSplit)
    n = floor(mcmc_iterSampleThin / nSplit)
    postDrawsSplit = np.zeros((m, n, nParam, nParam2))
    postDrawsSplit[0:mcmc_nChain, :, :, :] = postDraws[:, 0:n, :, :]
    postDrawsSplit[mcmc_nChain:m, :, :, :] = postDraws[:,n:mcmc_iterSampleThin, :, :]
    muChain = np.mean(postDrawsSplit, axis = 1)
    muChainArr = np.array(muChain).reshape((m,1,nParam, nParam2))
    mu = np.array(np.mean(muChain, axis = 0)).reshape((1, nParam, nParam2))
    B = (n / (m - 1)) * np.sum((muChain - mu)**2)
    sSq = (1 / (n - 1)) * np.sum((postDrawsSplit - muChainArr)**2, axis = 1)
    W = np.mean(sSq, axis = 0)
    varPlus = ((n - 1) / n) * W + B / n
    Rhat = np.empty((nParam, nParam2)) * np.nan
    W_idx = W > 0
    Rhat[W_idx] = np.sqrt(varPlus[W_idx] / W[W_idx])
    tabPostAna[:, 4] = np.array(Rhat).reshape((nParam * nParam2,))
    
    if paramName not in ["Omega", "Corr", "paramRnd"]:
        postMean = np.ndarray.flatten(postMean)
        
    pdTabPostAna = pd.DataFrame(tabPostAna, columns = colHeaders) 
    return postMean, pdTabPostAna             

###
#Estimate
###
    
def estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simDraws,
        rhoF, rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        paramFix_inits, zeta_inits, Omega_inits,
        indID, obsID, altID, chosen,
        xFix, xRnd,
        xFix_trans, xRnd_trans):
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    
    xFix_transBool = np.sum(xFix_trans) > 0
    xRnd_transBool = np.sum(xRnd_trans) > 0  
    
    xList = [xFix, xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd = xList[0], xList[1]
    
    ### 
    #Posterior sampling
    ###
    
    mcmc_iter = mcmc_iterBurn + mcmc_iterSample
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    mcmc_iterMemThin = floor(mcmc_iterMem / mcmc_thin)

    A = A * np.ones((nRnd,))
    invASq = A ** (-2)
    
    paramFix = paramFix_inits
    zeta = zeta_inits
    Omega = Omega_inits
    
    tic = time.time()

    for c in range(mcmc_nChain):
        mcmcChain(c, seed,
                mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
                rhoF, rho,    
                modelName,
                paramFix, zeta, Omega, invASq, nu, diagCov,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd, 
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs) 
    """
    Parallel(n_jobs = mcmc_nChain)(delayed(mcmcChain)(
                c, seed,
                mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
                rhoF, rho,    
                modelName,
                paramFix, zeta, Omega, invASq, nu, diagCov,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd, 
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs) 
    for c in range(mcmc_nChain))
    """

    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))
        
    ###
    #Posterior analysis
    ###

    if nFix > 0:        
        postMean_paramFix, pdTabPostAna_paramFix = postAna('paramFix', nFix, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Fixed parameters:')    
        print(pdTabPostAna_paramFix)
    else:
        postMean_paramFix = None; pdTabPostAna_paramFix = None;
 
    if nRnd > 0:
        postMean_zeta, pdTabPostAna_zeta = postAna('zeta', nRnd, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (means):')    
        print(pdTabPostAna_zeta)
        
        postMean_sd, pdTabPostAna_sd = postAna('sd', nRnd, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (standard deviations):')    
        print(pdTabPostAna_sd)
        
        postMean_Omega, pdTabPostAna_Omega = postAna('Omega', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (covariance matrix):')    
        print(pdTabPostAna_Omega)
        
        postMean_Corr, pdTabPostAna_Corr = postAna('Corr', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (correlation matrix):')    
        print(pdTabPostAna_Corr)
        
        postMean_paramRnd, pdTabPostAna_paramRnd = postAna('paramRnd', nInd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
    else:
        postMean_zeta = None; pdTabPostAna_zeta = None;
        postMean_sd = None; pdTabPostAna_sd = None;
        postMean_Omega = None; pdTabPostAna_Omega = None;
        postMean_Corr = None; pdTabPostAna_Corr = None;
        postMean_paramRnd = None; pdTabPostAna_paramRnd = None;
    
    ###
    #Simulate log-likelihood at posterior means
    ###
    
    if nFix > 0 and nRnd == 0:
        simDraws_star = 1
    else:
        simDraws_star = simDraws
    
    pSim = np.zeros((simDraws_star, nInd))
    
    paramFix = 0; paramRnd = 0;
    if nFix > 0: paramFix = postMean_paramFix
    if nRnd > 0: postMean_chOmega = np.linalg.cholesky(postMean_Omega)      
                
    for i in np.arange(simDraws_star):
        if nRnd > 0:
            paramRnd = postMean_zeta + (postMean_chOmega @ np.random.randn(nRnd, nInd)).T
            
        lPInd = probMxl(
                paramFix, paramRnd,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd,
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)
        pSim[i, :] = np.exp(lPInd)
    
    logLik = np.sum(np.log(np.mean(pSim, axis = 0)))
    print(' ')
    print('Log-likelihood (simulated at posterior means): ' + str(logLik)) 
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 
        
    ###
    #Save results
    ###
    
    results = {'modelName': modelName, 'seed': seed,
               'estimation_time': toc,
               'logLik': logLik,
               'postMean_paramFix': postMean_paramFix, 'pdTabPostAna_paramFix': pdTabPostAna_paramFix,
               'postMean_zeta': postMean_zeta, 'pdTabPostAna_zeta': pdTabPostAna_zeta, 
               'postMean_sd': postMean_sd, 'pdTabPostAna_sd': pdTabPostAna_sd, 
               'postMean_Omega': pdTabPostAna_Omega, 'pdTabPostAna_Omega': pdTabPostAna_Omega, 
               'postMean_Corr': postMean_Corr, 'pdTabPostAna_Corr': pdTabPostAna_Corr,
               'postMean_paramRnd': postMean_paramRnd, 'pdTabPostAna_paramRnd': pdTabPostAna_paramRnd
               }
    
    return results

###
#Prediction
###
    
def mcmcChainPred(
        chainID, seed,
        mcmc_iterSampleThin, mcmc_disp, nTakes, nSim,
        modelName,
        xFix, nFix, 
        sim_xRnd, nRnd, 
        nInd, nObs, nRow,
        sim_rowsPerInd, sim_map_avail_to_obs, chosenIdx, nonChosenIdx):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Retrieve draws
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    file = h5py.File(fileName, "r")
    
    paramFix_store = None
    if nFix: paramFix_store = np.array(file['paramFix_store'])
    zeta_store = np.array(file['zeta_store'])
    Omega_store = np.array(file['Omega_store'])
    
    ###
    #Simulate
    ###

    pPred = np.zeros((nRow + nObs,))
    vFix = 0 
    
    for i in np.arange(mcmc_iterSampleThin):
        
        if nFix: 
            paramFix = paramFix_store[i,:]
            vFix = np.tile(xFix @ paramFix, (nSim,));
        
        zeta_tmp = zeta_store[i,:]
        ch_tmp = np.linalg.cholesky(Omega_store[i,:,:])
        
        pPred_iter = np.zeros((nRow + nObs,))
        
        for t in np.arange(nTakes):
            paramRnd = zeta_tmp + (ch_tmp @ np.random.randn(nRnd, nInd * nSim)).T
            paramRndPerRow = np.repeat(paramRnd, sim_rowsPerInd, axis = 0)
            vRnd = np.sum(sim_xRnd * paramRndPerRow, axis = 1)
            
            pPred_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
            pPred_iter += pPred_take
            
        pPred += (pPred_iter / nTakes)
        
        if ((i + 1) % mcmc_disp) == 0:
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (predictive simulation)')
            sys.stdout.flush()
            
    pPred /= mcmc_iterSampleThin
    return pPred
    
def predict(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xFix, xRnd):
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    
    xList = [xFix, xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd = xList[0], xList[1]
    
    sim_xRnd = np.tile(xRnd, (nSim, 1))
    sim_rowsPerInd = np.tile(rowsPerInd, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)
    
    ### 
    #Predictive simulation
    ###
    
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    
    pPred = np.zeros((nObs + nRow,))
    for c in np.arange(mcmc_nChain):
        predPred_chain = mcmcChainPred(
                c, seed,
                mcmc_iterSampleThin, mcmc_disp, nTakes, nSim,
                modelName,
                xFix, nFix, 
                sim_xRnd, nRnd, 
                nInd, nObs, nRow,
                sim_rowsPerInd, sim_map_avail_to_obs, chosenIdx, nonChosenIdx)
        pPred += predPred_chain
    pPred /= mcmc_nChain
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 

    return pPred

###
#If main: test
###
    
if __name__ == "__main__":
    
    import random
    random.seed(4711)
    
    ###
    #Load data
    ###
    
    data = pd.read_csv('swissmetro_long.csv')
    data = data[((data['PURPOSE'] != 1) & (data['PURPOSE'] != 3)) != True]
    
    ###
    #Prepare data
    ###
    
    indID = np.array(data['ID'].values, dtype = 'int64')
    obsID = np.array(data['obsID'].values, dtype = 'int64')
    altID = np.array(data['altID'].values, dtype = 'int64')
    
    chosen = np.array(data['chosen'].values, dtype = 'int64')
    
    tt = np.array(data['TT'].values, dtype = 'float64') / 10
    cost = np.array(data['CO'].values, dtype = 'float64') / 10
    he = np.array(data['HE'].values, dtype = 'float64') / 10
    ga = np.array(data['GA'].values, dtype = 'int64')
    cost[(altID <= 2) & (ga == 1)] = 0
    
    const2 = 1 * (altID == 2)
    const3 = 1 * (altID == 3)
    
    ###
    #Estimate MXL via MCMC
    ###
    
    xFix = np.stack((const2, const3), axis = 1)
    xRnd = -np.stack((cost, tt), axis = 1) #np.zeros((0,0)) #-np.hstack((cost, he, tt))
    
    #Fixed parameter distributions
    #0: normal
    #1: log-normal (to assure that fixed parameter is striclty negative or positive)
    xFix_trans = np.array([0, 0])
    
    #Random parameter distributions
    #0: normal
    #1: log-normal
    #2: S_B
    xRnd_trans = np.array([0, 0])
    
    paramFix_inits = np.zeros((xFix.shape[1],))
    zeta_inits = np.zeros((xRnd.shape[1],))
    Omega_inits = 0.1 * np.eye(xRnd.shape[1])
    
    A = 1.04
    nu = 2
    diagCov = False
    
    mcmc_nChain = 2
    mcmc_iterBurn = 20000
    mcmc_iterSample = 20000
    mcmc_thin = 5
    mcmc_iterMem = 20000
    mcmc_disp = 1000
    seed = 4711
    simDraws = 1000    
    
    rho = 0.1
    rhoF = 0.01
    
    modelName = 'test'
    deleteDraws = True
    
    results = estimate(
            mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
            seed, simDraws,
            rhoF, rho,
            modelName, deleteDraws,
            A, nu, diagCov,
            paramFix_inits, zeta_inits, Omega_inits,
            indID, obsID, altID, chosen,
            xFix, xRnd,
            xFix_trans, xRnd_trans)