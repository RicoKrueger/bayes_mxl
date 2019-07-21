#import os
#import sys
import time
import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.stats

from mxl import prepareData, transFix, transRnd
from qmc import makeNormalDraws
                
###
#MSLE
###

def transDerFix(derFix, paramFix, xFix_trans):
    derFix_trans = np.array(derFix)
    
    idx = xFix_trans == 1
    if np.sum(idx) > 0:
        derFix_trans[:, idx] = derFix[:, idx] * paramFix[idx]
    return derFix_trans
    
def transDerRnd(derRnd, paramRnd, xRnd_trans):
    derRnd_trans = np.array(derRnd)
    
    idx = xRnd_trans == 1
    if np.sum(idx) > 0:
        derRnd_trans[:, idx] = paramRnd[:, idx]
        
    idx = xRnd_trans == 2
    if np.sum(idx) > 0:
        derRnd_trans[:, idx] = paramRnd[:, idx] - paramRnd[:, idx] ** 2    
    return derRnd_trans

def derRnd(nInd, nDrawsMem, nRnd, xRnd_transBool, paramRnd, xRnd_trans, chIdx, 
           drawsTake, sim_rowsPerInd, uc):
    derRnd = np.ones((nInd * nDrawsMem, nRnd))
    if xRnd_transBool: derRnd = transDerRnd(derRnd, paramRnd, xRnd_trans)
    derRnd_mu_ind = derRnd
    if uc:
        derRnd_ch_ind = derRnd * drawsTake
    else:
        derRnd_ch_ind = derRnd[:, chIdx[0]] * drawsTake[:, chIdx[1]]
    derRnd_mu = np.repeat(derRnd_mu_ind, sim_rowsPerInd, axis = 0)
    derRnd_ch = np.repeat(derRnd_ch_ind, sim_rowsPerInd, axis = 0)
    return derRnd_mu, derRnd_ch

def probGrMxl(
        param,
        sim_xFix, xFix_transBool, xFix_trans, nFix, 
        sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
        sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
        drawsUcTake, drawsCoTake, nDrawsMem,
        nInd, sim_rowsPerInd, sim_map_obs_to_ind, sim_map_avail_to_obs, 
        sim_map_ind_to_avail, sim_map_draws_to_ind):
    ###
    #Utility
    ###
    
    vFix = 0; vRndUc = 0; vRndCo = 0;
    u = 0
    if nFix > 0:
        l = u; u += nFix;
        paramFix = np.array(param[l:u])
        if xFix_transBool: paramFix = np.array(transFix(paramFix, xFix_trans))
        vFix = sim_xFix @ paramFix
    if nRndUc > 0:
        l = u; u += nRndUc;
        paramRndUc_mu = np.array(param[l:u])
        l = u; u += nRndUc;
        paramRndUc_sd = np.array(param[l:u])
        paramRndUc = paramRndUc_mu + paramRndUc_sd * drawsUcTake
        if xRndUc_transBool: paramRndUc = np.array(transRnd(paramRndUc, xRndUc_trans))
        paramRndUcPerRow = np.repeat(paramRndUc, sim_rowsPerInd, axis = 0)
        vRndUc = np.sum(sim_xRndUc * paramRndUcPerRow, axis = 1)       
    if nRndCo > 0:
        l = u; u += nRndCo;
        paramRndCo_mu = np.array(param[l:u])
        l = u
        paramRndCo_ch = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch[chIdx] = np.array(param[l:])
        paramRndCo = paramRndCo_mu + (paramRndCo_ch @ drawsCoTake.T).T
        if xRndCo_transBool: paramRndCo = np.array(transRnd(paramRndCo, xRndCo_trans))
        paramRndCoPerRow = np.repeat(paramRndCo, sim_rowsPerInd, axis = 0)
        vRndCo = np.sum(sim_xRndCo * paramRndCoPerRow, axis = 1)
        
    v = vFix + vRndUc + vRndCo
    
    ###
    #Probability
    ###
    
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-200] = 1e-200 
    nev = sim_map_avail_to_obs.T @ ev + 1
    nnev = sim_map_avail_to_obs * nev;
    pChosen = 1 / nev
    pChosen[pChosen < 1e-200] = 1e-200
    pNonChosen = ev / nnev
    pNonChosen[pNonChosen < 1e-200] = 1e-200
    lPChosen = np.log(pChosen)
    lPInd = sim_map_obs_to_ind.T @ lPChosen
    pIndVec = np.exp(lPInd)
    pInd = pIndVec.reshape((nDrawsMem, nInd)).sum(axis = 0)
    
    ###
    #Gradient
    ###
    
    def calcGradient(der):
        frac = -pNonChosen.reshape((-1,1)) * der
        summation = sim_map_ind_to_avail @ frac
        prod = pIndVec.reshape((-1,1)) * summation
        sgr = sim_map_draws_to_ind @ prod
        return sgr
    
    sgrFix = np.empty((nInd, 0))
    sgrRndUc_mu = np.empty((nInd, 0))
    sgrRndUc_sd = np.empty((nInd, 0))
    sgrRndCo_mu = np.empty((nInd, 0))
    sgrRndCo_ch = np.empty((nInd, 0))
    
    if nFix > 0:
        derFix = sim_xFix
        if xFix_transBool: derFix = np.array(transDerFix(derFix, paramFix, xFix_trans))
        sgrFix = calcGradient(derFix)
    
    if nRndUc > 0:
        derRndUc_mu, derRndUc_sd = derRnd(nInd, nDrawsMem, nRndUc, 
                                          xRndUc_transBool, paramRndUc, xRndUc_trans, 
                                          chIdx, drawsUcTake, sim_rowsPerInd, True)
        derRndUc_mu *= sim_xRndUc
        derRndUc_sd *= sim_xRndUc
        sgrRndUc_mu = calcGradient(derRndUc_mu)
        sgrRndUc_sd = calcGradient(derRndUc_sd)    
    
    if nRndCo > 0:
        derRndCo_mu, derRndCo_ch = derRnd(nInd, nDrawsMem, nRndCo, 
                                          xRndCo_transBool, paramRndCo, xRndCo_trans, 
                                          chIdx, drawsCoTake, sim_rowsPerInd, False)
        derRndCo_mu *= sim_xRndCo
        derRndCo_ch *= sim_xRndCo[:, chIdx[0]]
        sgrRndCo_mu = calcGradient(derRndCo_mu)
        sgrRndCo_ch = calcGradient(derRndCo_ch)    
 
    sgr = np.concatenate((sgrFix, 
                          sgrRndUc_mu, sgrRndUc_sd, 
                          sgrRndCo_mu, sgrRndCo_ch), axis = 1) 
    
    return pInd, sgr

def objectiveMxl(
        param,
        sim_xFix, xFix_transBool, xFix_trans, nFix, 
        sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
        sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
        drawsUc, drawsCo, nDraws, nDrawsMem, nTakes,
        nInd, sim_rowsPerInd, sim_map_obs_to_ind, sim_map_avail_to_obs, sim_map_ind_to_avail, sim_map_draws_to_ind):
    pIndSim = np.zeros((nInd,))
    sgrSim = np.zeros((nInd, param.shape[0]));
    lengthMem = nInd * nDrawsMem; b = 0;
    for t in np.arange(nTakes):
        if nTakes > 0:
            a = b; b += lengthMem; ab = slice(a,b);
            if nRndUc > 0:
                drawsUcTake = drawsUc[ab, :]
            else:
                drawsUcTake = drawsUc  
            if nRndCo > 0:
                drawsCoTake = drawsCo[ab, :]    
            else:
                drawsCoTake = drawsCo               
        
        pIndTake, sgrTake = probGrMxl(
                param,
                sim_xFix, xFix_transBool, xFix_trans, nFix, 
                sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
                sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
                drawsUcTake, drawsCoTake, nDrawsMem,
                nInd, sim_rowsPerInd, sim_map_obs_to_ind, 
                sim_map_avail_to_obs, sim_map_ind_to_avail, sim_map_draws_to_ind)
        
        pIndSim += pIndTake
        sgrSim += sgrTake
        
    sgrSim /= pIndSim.reshape((-1,1))
    pIndSim /= nDraws
    
    ll = -np.sum(np.log(pIndSim), axis = 0)
    gr = -np.sum(sgrSim, axis = 0)
    return ll, gr

###
#Process output
###
 
def processOutput(est, se, zVal, pVal, lu):
    colHeaders = ['est.', 'std. err.', 'z-val.', 'p-val.']
    param_est = est[lu]
    param_se = se[lu]
    param_zVal = zVal[lu]
    param_pVal = pVal[lu]
    pd_param = pd.DataFrame(np.stack((param_est, param_se, param_zVal, param_pVal), axis = 1), columns = colHeaders)
    print(pd_param)
    return param_est, param_se, param_zVal, param_pVal, pd_param

###
#Conditional expectation of individual-specific parameters
###
    
def probMxl(v, map_obs_to_ind, map_avail_to_obs):
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    nev = map_avail_to_obs.T @ ev + 1
    pChosen = 1 / nev
    lPChosen = np.log(pChosen)
    lPInd = map_obs_to_ind.T @ lPChosen
    pInd = np.exp(lPInd)
    return pInd

def condExpInd(
        paramFix_est, 
        paramRndUc_mu_est, paramRndUc_sd_est,
        paramRndCo_mu_est, paramRndCo_ch_est,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
        xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs, nSim):
    
    if nRndUc: paramRndUc_draws = np.zeros((nSim, nInd, nRndUc))
    if nRndCo: paramRndCo_draws = np.zeros((nSim, nInd, nRndCo))
    pInd_draws = np.zeros((nSim, nInd))
    
    vFix = 0; vRndUc = 0; vRndCo = 0;
    
    if nFix > 0: vFix = xFix @ paramFix_est
        
    for i in np.arange(nSim):
        if nRndUc > 0:
            paramRndUc = paramRndUc_mu_est + paramRndUc_sd_est * np.random.randn(nInd, nRndUc)
            paramRndUc_draws[i,:,:] = paramRndUc
            if xRndUc_transBool: paramRndUc = np.array(transRnd(paramRndUc, xRndUc_trans))
            paramRndUcPerRow = np.repeat(paramRndUc, rowsPerInd, axis = 0)
            vRndUc = np.sum(xRndUc * paramRndUcPerRow, axis = 1)
            
        if nRndCo > 0:
            paramRndCo = paramRndCo_mu_est + (paramRndCo_ch_est @ np.random.randn(nRndCo, nInd)).T
            paramRndCo_draws[i,:,:] = paramRndCo
            if xRndCo_transBool: paramRndCo = np.array(transRnd(paramRndCo, xRndCo_trans))
            paramRndCoPerRow = np.repeat(paramRndCo, rowsPerInd, axis = 0)
            vRndCo = np.sum(xRndCo * paramRndCoPerRow, axis = 1)
            
        v = vFix + vRndUc + vRndCo
        pInd_draws[i, :] = probMxl(v, map_obs_to_ind, map_avail_to_obs)
    
    denom = np.mean(pInd_draws, axis = 0)
    pInd_draws = np.array(pInd_draws).reshape((nSim, nInd, 1))
    
    paramRndUc_ind = None; paramRndCo_ind = None;
    
    if nRndUc > 0:
        numer = np.mean(paramRndUc_draws * pInd_draws, axis = 0)
        paramRndUc_ind = numer / denom.reshape((nInd, 1)) 
    if nRndCo > 0:
        numer = np.mean(paramRndCo_draws * pInd_draws, axis = 0)
        paramRndCo_ind = numer / denom.reshape((nInd, 1)) 
        
    return paramRndUc_ind, paramRndCo_ind

###
#Estimate
###
    
def estimate(
        drawsType, nDraws, nTakes, seed, modelName, deleteDraws,
        simCondInd, nSim,
        paramFix_inits, paramRndUc_mu_inits, paramRndUc_sd_inits, 
        paramRndCo_mu_inits, paramRndCo_ch_inits,
        indID, obsID, altID, chosen,
        xFix, xRndUc, xRndCo,
        xFix_trans, xRndUc_trans, xRndCo_trans):
    
    np.random.seed(seed)
    
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRndUc = xRndUc.shape[1]
    nRndCo = xRndCo.shape[1]
    nRnd = nRndUc + nRndCo
    
    if nRnd > 0:
        nDrawsMem, mod = divmod(nDraws, nTakes)
        assert mod == 0, "nDraws is not multiple of nTakes!"
    else:
        nDraws, nDrawsMem, nTakes = 1, 1, 1
    
    xFix_transBool = np.sum(xFix_trans) > 0
    xRndUc_transBool = np.sum(xRndUc_trans) > 0 
    xRndCo_transBool = np.sum(xRndCo_trans) > 0 
    
    xList = [xFix, xRndUc, xRndCo]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRndUc, xRndCo = xList[0], xList[1], xList[2]
    
    sim_xFix, sim_xRndUc, sim_xRndCo = np.tile(xFix, (nDrawsMem, 1)),np.tile(xRndUc, (nDrawsMem, 1)), np.tile(xRndCo, (nDrawsMem, 1))
    sim_rowsPerInd = np.tile(rowsPerInd, (nDrawsMem,))
    sim_map_obs_to_ind = scipy.sparse.kron(scipy.sparse.eye(nDrawsMem), map_obs_to_ind)
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nDrawsMem), map_avail_to_obs)
    sim_map_draws_to_ind = scipy.sparse.hstack([scipy.sparse.eye(nInd) for i in np.arange(nDrawsMem)])
    sim_map_ind_to_avail = (sim_map_avail_to_obs @ sim_map_obs_to_ind).T
    
    chIdx = None
    if nRndCo: 
        chIdx = np.triu_indices(nRndCo); chIdx = chIdx[1], chIdx[0];
             
    ### 
    #Generate draws
    ###
    
    drawsUc = None; drawsCo = None;
    if nRndUc: _, drawsUc = makeNormalDraws(nDraws, nRndUc, drawsType, nInd)
    if nRndCo: _, drawsCo = makeNormalDraws(nDraws, nRndCo, drawsType, nInd)   
    
    ### 
    #Optimise
    ###
    
    paramRndCo_chVec_inits = np.ndarray.flatten(paramRndCo_ch_inits[chIdx])
    inits = np.concatenate((paramFix_inits, 
                            paramRndUc_mu_inits, paramRndUc_sd_inits, 
                            paramRndCo_mu_inits, paramRndCo_chVec_inits), axis = 0)
    
    tic = time.time()
    resOpt = sp.optimize.minimize(
            fun = objectiveMxl,
            x0 = inits,
            args = (sim_xFix, xFix_transBool, xFix_trans, nFix, 
                    sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
                    sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
                    drawsUc, drawsCo, nDraws, nDrawsMem, nTakes,
                    nInd, sim_rowsPerInd, sim_map_obs_to_ind, 
                    sim_map_avail_to_obs, sim_map_ind_to_avail, sim_map_draws_to_ind),
            method = 'BFGS',
            jac = True,
            options = {'disp': True})
    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))
    
    ###
    #Process output
    ###
    
    logLik = -resOpt['fun']
    est = resOpt['x']
    iHess = resOpt['hess_inv']
    se = np.sqrt(np.diag(iHess))
    zVal = est / se
    pVal = 2 * scipy.stats.norm.cdf(-np.absolute(zVal))

    u = 0
    if nFix > 0:
        l = u; u += nFix; lu = slice(l,u)
        print(' ')
        print('Fixed parameters:')
        paramFix_est, paramFix_se, paramFix_zVal, paramFix_pVal, pd_paramFix = processOutput(est, se, zVal, pVal, lu)
    else:
        paramFix_est, paramFix_se, paramFix_zVal, paramFix_pVal, pd_paramFix = None, None, None, None, None
        
    if nRndUc > 0:
        l = u; u += nRndUc; lu = slice(l,u)
        print(' ')
        print('Uncorrelated random parameters (means):')
        paramRndUc_mu_est, paramRndUc_mu_se, paramRndUc_mu_zVal, paramRndUc_mu_pVal, pd_paramRndUc_mu = processOutput(est, se, zVal, pVal, lu)
        
        l = u; u += nRndUc; lu = slice(l,u)
        print(' ')
        print('Uncorrelated random parameters (standard deviations):')
        paramRndUc_sd_est, paramRndUc_sd_se, paramRndUc_sd_zVal, paramRndUc_sd_pVal, pd_paramRndUc_sd = processOutput(est, se, zVal, pVal, lu) 
    else:
        paramRndUc_mu_est, paramRndUc_mu_se, paramRndUc_mu_zVal, paramRndUc_mu_pVal, pd_paramRndUc_mu = None, None, None, None, None
        paramRndUc_sd_est, paramRndUc_sd_se, paramRndUc_sd_zVal, paramRndUc_sd_pVal, pd_paramRndUc_sd = None, None, None, None, None
            
    if nRndCo > 0:
        l = u; u += nRndCo; lu = slice(l,u)
        print(' ')
        print('Correlated random parameters (means):')        
        paramRndCo_mu_est, paramRndCo_mu_se, paramRndCo_mu_zVal, paramRndCo_mu_pVal, pd_paramRndCo_mu = processOutput(est, se, zVal, pVal, lu)
        
        l = u; u = est.shape[0]; lu = slice(l,u)
        print(' ')
        print('Correlated random parameters (Cholesky):')
        paramRndCo_ch_est_vec, paramRndCo_ch_se_vec, paramRndCo_ch_zVal_vec, paramRndCo_ch_pVal_vec, pd_paramRndCo_ch = processOutput(est, se, zVal, pVal, lu) 

        print(' ')
        print('Correlated random parameters (Cholesky, est.):')
        paramRndCo_ch_est = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch_est[chIdx] = paramRndCo_ch_est_vec
        print(pd.DataFrame(paramRndCo_ch_est))
        
        print(' ')
        print('Correlated random parameters (Cholesky, std. err.):')
        paramRndCo_ch_se = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch_se[chIdx] = paramRndCo_ch_se_vec
        print(pd.DataFrame(paramRndCo_ch_se))
        
        print(' ')
        print('Correlated random parameters (Cholesky, p-val.):')
        paramRndCo_ch_pVal = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch_pVal[chIdx] = paramRndCo_ch_pVal_vec  
        print(pd.DataFrame(paramRndCo_ch_pVal))
    else:
        paramRndCo_mu_est, paramRndCo_mu_se, paramRndCo_mu_zVal, paramRndCo_mu_pVal, pd_paramRndCo_mu = None, None, None, None, None
        paramRndCo_ch_est, paramRndCo_ch_se, paramRndCo_ch_pVal, pd_paramRndCo_ch = None, None, None, None
        
    print(' ')
    print('Log-likelihood: ' + str(logLik)) 
    print(' ')
    
    if nRnd:
        print('QMC method: ' + drawsType)
        print('Number of simulation draws: ' + str(nDraws))
    
    ###
    #Conditional expectation of individual-specific parameters
    ###
    
    if simCondInd and nRnd > 0:
        paramRndUc_ind, paramRndCo_ind = condExpInd(
                paramFix_est, 
                paramRndUc_mu_est, paramRndUc_sd_est,
                paramRndCo_mu_est, paramRndCo_ch_est,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
                xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo,
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs, nSim)
    else:
        paramRndUc_ind, paramRndCo_ind = None, None

    ###
    #Delete draws
    ###      
    
    if deleteDraws:
        drawsUc = None; drawsCo = None; 
    
    ###
    #Save results
    ###
    
    results = {'modelName': modelName, 'seed': seed,
               'estimation_time': toc, 'drawsType': drawsType, 'nDraws': nDraws,
               'nSim': nSim,
               'drawsUc': drawsUc, 'drawsCo': drawsCo,
               'logLik': logLik, 'est': est, 'iHess': iHess,
               'paramFix_est': paramFix_est, 'paramFix_se': paramFix_se, 'paramFix_zVal': paramFix_zVal, 'paramFix_pVal': paramFix_pVal, 'pd_paramFix': pd_paramFix,
               'paramRndUc_mu_est': paramRndUc_mu_est, 'paramRndUc_mu_se': paramRndUc_mu_se, 'paramRndUc_mu_zVal': paramRndUc_mu_zVal, 'paramRndUc_mu_pVal': paramRndUc_mu_pVal, 'pd_paramRndUc_mu': pd_paramRndUc_mu,
               'paramRndUc_sd_est': paramRndUc_sd_est, 'paramRndUc_sd_se': paramRndUc_sd_se, 'paramRndUc_sd_zVal': paramRndUc_sd_zVal, 'paramRndUc_sd_pVal': paramRndUc_sd_pVal, 'pd_paramRndUc_sd': pd_paramRndUc_sd,
               'paramRndCo_mu_est': paramRndCo_mu_est, 'paramRndCo_mu_se': paramRndCo_mu_se, 'paramRndCo_mu_zVal': paramRndCo_mu_zVal, 'paramRndCo_mu_pVal': paramRndCo_mu_pVal, 'pd_paramRndCo_mu': pd_paramRndCo_mu,
               'paramRndCo_ch_est': paramRndCo_ch_est, 'paramRndCo_ch_se': paramRndCo_ch_se, 'paramRndCo_ch_pVal': paramRndCo_ch_pVal, 'pd_paramRndCo_ch': pd_paramRndCo_ch,
               'paramRndUc_ind': paramRndUc_ind, 'paramRndCo_ind': paramRndCo_ind,
               'resOpt': resOpt
               }
        
    return results

###
#If main: test
###
    
if __name__ == "__main__":
    
    np.random.seed(4711)
    
    """
    ###
    #Load data
    ###
    
    data = pd.read_csv('swissmetro_long.csv')
    data = data[((data['PURPOSE'] != 1) & (data['PURPOSE'] != 3)) != True]
    data = data[data['ID'] <= 200]
    
    ###
    #Prepare data
    ###
    
    indID = np.array(data['indID'].values, dtype = 'int64')
    obsID = np.array(data['obsID'].values, dtype = 'int64')
    altID = np.array(data['altID'].values, dtype = 'int64')
    
    chosen = np.array(data['chosen'].values, dtype = 'int64')
    
    tt = np.array(data['TT'].values, dtype = 'float64') / 10
    cost = np.array(data['CO'].values, dtype = 'float64') / 10
    he = np.array(data['HE'].values, dtype = 'float64')/ 10
    ga = np.array(data['GA'].values, dtype = 'int64')
    cost[(altID <= 2) & (ga == 1)] = 0
    
    const2 = 1 * (altID == 2)
    const3 = 1 * (altID == 3)
    """
    ###
    #Generate data
    ###
    
    N = 1000
    T = 1
    NT = N * T
    J = 5
    NTJ = NT * J
    
    L = 3 #no. of fixed paramters
    K = 5 #no. of random parameters
    
    true_alpha = np.array([-0.8, 0.8, 1.2])
    true_zeta = np.array([-0.8, 0.8, 1.0, -0.8, 1.5])
    true_Omega = np.array([[1.0, 0.8, 0.8, 0.8, 0.8],
                           [0.8, 1.0, 0.8, 0.8, 0.8],
                           [0.8, 0.8, 1.0, 0.8, 0.8],
                           [0.8, 0.8, 0.8, 1.0, 0.8],
                           [0.8, 0.8, 0.8, 0.8, 1.0]])
    
    xFix = 0 * np.random.rand(NTJ, L)
    xRnd = np.random.rand(NTJ, K)

    betaInd_tmp = true_zeta + \
    (np.linalg.cholesky(true_Omega) @ np.random.randn(K, N)).T
    beta_tmp = np.kron(betaInd_tmp, np.ones((T * J,1)))
    
    eps = -np.log(-np.log(np.random.rand(NTJ,)))
    
    vDet = xFix @ true_alpha + np.sum(xRnd * beta_tmp, axis = 1)
    v = vDet + eps
    
    vDetMax = np.zeros((NT,))
    vMax = np.zeros((NT,))
    
    chosen = np.zeros((NTJ,), dtype = 'int64')
    
    for t in np.arange(NT):
        l = t * J; u = (t + 1) * J
        altMaxDet = np.argmax(vDet[l:u])
        altMax = np.argmax(v[l:u])
        vDetMax[t] = altMaxDet
        vMax[t] = altMax
        chosen[l + altMax] = 1
        
    error = np.sum(vMax == vDetMax) / NT * 100
    print(error)
    
    indID = np.repeat(np.arange(N), T * J)
    obsID = np.repeat(np.arange(NT), J)
    altID = np.tile(np.arange(J), NT)    
    
    ###
    #Estimate MXL via MSLE
    ###
    
    xFix = np.zeros((0,0)) #np.stack((const2, const3), axis = 1)
    xRndUc = np.zeros((0,0)) # #-np.hstack((cost, he, tt))
    xRndCo = xRnd.copy() #np.stack((cost, he, tt), axis = 1)
    
    #Fixed parameter distributions
    #0: normal
    #1: log-normal (to assure that fixed parameter is striclty negative or positive)
    xFix_trans = np.array([0, 0, 0, 0])
    
    #Random parameter distributions
    #0: normal
    #1: log-normal
    #2: S_B
    xRndUc_trans = np.array([0, 0])
    xRndCo_trans = np.array([0, 0])
    
    paramFix_inits = np.zeros((xFix.shape[1],))
    paramRndUc_mu_inits = np.zeros((xRndUc.shape[1],))
    paramRndUc_sd_inits = np.ones((xRndUc.shape[1],))
    paramRndCo_mu_inits = np.zeros((xRndCo.shape[1],))
    paramRndCo_ch_inits = 0.1 * np.eye(xRndCo.shape[1])
    
    drawsType = 'mlhs'
    nDraws = 1000
    nTakes = 2
    seed = 4711
    
    simCondInd = True
    nSim = 1000

    modelName = 'test'
    deleteDraws = True
    
    results = estimate(
            drawsType, nDraws, nTakes, seed, modelName, deleteDraws,
            simCondInd, nSim,
            paramFix_inits, paramRndUc_mu_inits, paramRndUc_sd_inits, paramRndCo_mu_inits, paramRndCo_ch_inits,
            indID, obsID, altID, chosen,
            xFix, xRndUc, xRndCo,
            xFix_trans, xRndUc_trans, xRndCo_trans)    