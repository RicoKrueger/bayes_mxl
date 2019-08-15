#import os
#import sys
import time
import pandas as pd
import numpy as np
import scipy as sp
from scipy.linalg import block_diag
import scipy.sparse
from scipy.stats import invwishart

from mxl import prepareData, pPredMxl
from qmc import makeNormalDraws
                
###
#E-LSE
###

#Modified Jensen's inequality
def mjiParamX(
        paramFixMu, paramFixSi,
        paramRndMu, paramRndSi,
        xFix, nFix,
        xRnd, nRnd,
        nInd):
    if nFix > 0 and nRnd == 0:
        paramMu = np.repeat(paramFixMu.reshape(1, nFix), nInd, axis = 0)
        paramSi = np.repeat(paramFixSi.reshape(1, nFix, nFix), nInd, axis = 0)
        xAll = xFix
    elif nFix == 0 and nRnd > 0:
        paramMu = paramRndMu
        paramSi = paramRndSi
        xAll = xRnd
    elif nFix > 0 and nRnd > 0:
        paramMu = np.zeros((nInd, nFix + nRnd))
        paramSi = np.zeros((nInd, nFix + nRnd, nFix + nRnd))
        for n in np.arange(nInd):
            paramMu[n,:] = np.concatenate((paramFixMu, paramRndMu[n,:]))
            paramSi[n,:,:] = block_diag(paramFixSi, paramRndSi[n,:,:])
        xAll = np.hstack((xFix, xRnd))
    return paramMu, paramSi, xAll

def nextMjiAux(
        paramFixMu, paramFixSi,
        paramRndMu, paramRndSi,
        mjiAux,
        xFix, nFix,
        xRnd, nRnd,
        nInd, nRow,
        map_avail, slIndRow):
    
    paramMu, paramSi, xAll = mjiParamX(
            paramFixMu, paramFixSi,
            paramRndMu, paramRndSi,
            xFix, nFix,
            xRnd, nRnd,
            nInd)    
    
    numer = np.zeros((nRow,))
    for n in np.arange(nInd):
        paramMu_n = paramMu[n,:]
        paramSi_n = paramSi[n,:,:] 
        a_n = mjiAux[slIndRow[n],:]
        x_n = xAll[slIndRow[n],:]
        map_avail_n = map_avail[slIndRow[n], slIndRow[n]]
        
        t1 = x_n @ paramMu_n
        t2 = 0.5 * (x_n - 2 * map_avail_n @ (a_n * x_n))
        t1 += np.diag(t2 @ paramSi_n @ x_n.T)
        numer[slIndRow[n]] = np.exp(t1)
        
    denom = map_avail @ numer
    mjiAux = np.array(numer / denom).reshape((nRow, 1))
    return mjiAux

def elseMjiFix(
        method,
        paramFixMu, paramFixCh, paramFixSi,
        paramRndMu, paramRndCh, paramRndSi,
        mjiAux,
        xFix, nFix, chFixIdx,
        xRnd, nRnd,
        nInd, nRow, rowsPerInd,
        map_avail_to_obs, slIndObs, slIndRow):  
    
    paramMu, paramSi, xAll = mjiParamX(
            paramFixMu, paramFixSi,
            paramRndMu, paramRndSi,
            xFix, nFix,
            xRnd, nRnd,
            nInd)

    lse = 0
    grFixMu = 0; grFixCh = 0; grFixSi = 0;
    nFixCh = int(nFix * (nFix + 1) / 2)
    
    for n in np.arange(nInd):
        paramMu_n = paramMu[n,:]
        paramSi_n = paramSi[n,:,:] 
        a_n = mjiAux[slIndRow[n],:]
        x_n = xAll[slIndRow[n],:]
        map_avail_to_obs_n = map_avail_to_obs[slIndRow[n], slIndObs[n]]
           
        t0 = map_avail_to_obs_n.T @ (a_n * x_n)
        t1 = t0 @ paramMu_n
        t2 = x_n - map_avail_to_obs_n @ t0
        t3 = t2 @ paramMu_n + 0.5 * np.diag(t2 @ paramSi_n @ t2.T)
        t3Max = np.max(t3)
        t4 = np.exp(t3 - t3Max)
        t5 = map_avail_to_obs_n.T @ t4
        lse -= np.sum(t1 + t3Max + np.log(t5))
  
        grFixMu += np.sum(-t0[:,:nFix], axis = 0)
        t6 = map_avail_to_obs_n @ t5
        t7 = -np.array(t4 / t6).reshape((-1,1))
        grFixMu += np.sum(t7 * t2[:,:nFix], axis = 0)
        if method[0] == 'qn': 
            t10 = np.zeros((rowsPerInd[n], nFixCh))
            for i in np.arange(rowsPerInd[n]):
                t8 = t2[i,:nFix]
                t9 = np.outer(t8, t8) @ paramFixCh
                t10[i,:] = t9[chFixIdx]
            grFixCh += np.sum(t7 * t10, axis = 0)     
        elif method[0] == 'ncvmp': 
            t10 = np.zeros((rowsPerInd[n], nFix * nFix))
            for i in np.arange(rowsPerInd[n]):
                t9 = t2[i,:nFix]
                t10[i,:] = np.outer(t9, t9).reshape((nFix * nFix,), order = 'F')
            grFixSi += 0.5 * np.sum(t7 * t10, axis = 0)         
    if method[0] == 'ncvmp': grFixSi = grFixSi.reshape((nFix, nFix), order = 'F')
    return lse, grFixMu, grFixCh, grFixSi

def elseMjiRnd(
        method,
        paramFixMu, paramFixSi,
        paramRndMu_n, paramRndCh_n, paramRndSi_n,
        a_n,
        xFix_n, nFix, 
        xRnd_n, nRnd, chRndIdx,
        nRow_n,
        map_avail_to_obs_n):  
    
    if nFix == 0:
        paramMu_n = paramRndMu_n
        paramSi_n = paramRndSi_n
        x_n = xRnd_n
    elif nFix > 0:
        paramMu_n = np.concatenate((paramFixMu, paramRndMu_n))
        paramSi_n = block_diag(paramFixSi, paramRndSi_n)
        x_n = np.hstack((xFix_n, xRnd_n))

    lse = 0
    grRndMu = 0; grRndCh = 0; grRndSi = 0;
    nRndCh = int(nRnd * (nRnd + 1) / 2)
           
    t0 = map_avail_to_obs_n.T @ (a_n * x_n)
    t1 = t0 @ paramMu_n
    t2 = x_n - map_avail_to_obs_n @ t0
    t3 = t2 @ paramMu_n + 0.5 * np.diag(t2 @ paramSi_n @ t2.T)
    t3Max = np.max(t3)
    t4 = np.exp(t3 - t3Max)
    t5 = map_avail_to_obs_n.T @ t4
    lse -= np.sum(t1 + t3Max + np.log(t5))
  
    grRndMu += np.sum(-t0[:,nFix:], axis = 0)
    t6 = map_avail_to_obs_n @ t5
    t7 = -np.array(t4 / t6).reshape((-1,1))
    grRndMu += np.sum(t7 * t2[:,nFix:], axis = 0)
    if method[0] == 'qn': 
        t10 = np.zeros((nRow_n, nRndCh))
        for i in np.arange(nRow_n):
            t8 = t2[i,nFix:]
            t9 = np.outer(t8, t8) @ paramRndCh_n
            t10[i,:] = t9[chRndIdx]
        grRndCh += np.sum(t7 * t10, axis = 0)     
    elif method[0] == 'ncvmp': 
        t10 = np.zeros((nRow_n, nRnd * nRnd))
        for i in np.arange(nRow_n):
            t8 = t2[i,nFix:]
            t10[i,:] = np.outer(t8, t8).reshape((nRnd * nRnd,), order = 'F')
        grRndSi += 0.5 * np.sum(t7 * t10, axis = 0)         
    if method[0] == 'ncvmp': grRndSi = grRndSi.reshape((nRnd, nRnd), order = 'F')
    return lse, grRndMu, grRndCh, grRndSi    

#Delta
def elseDeltaFix(
        method,
        paramFixMu, paramFixCh, paramFixSi, 
        paramRndSi,
        vFix, vRnd, 
        xFix, nFix, chFixIdx,
        xRnd, nRnd,
        nInd, nRow, obsPerInd,
        map_avail_to_obs, map_avail, slObsInd, slObsRow):    
    v = vFix + vRnd
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300
    nev = map_avail_to_obs.T @ ev
    nnev = map_avail_to_obs @ nev   
    p = ev / nnev; pVec = p.reshape((nRow,1));
    
    lse = -np.sum(np.log(nev))
    pProdX = pVec * xFix
    grFixMu = -np.sum(pProdX, axis = 0)
    grFixCh = 0; grFixSi = 0;
    hessFix = 0; hessRnd = 0;
    dP = pProdX - pVec * (map_avail @ pProdX)
    paramRndSi_n = None
    
    k = -1
    for n in np.arange(nInd):
        if nRnd: paramRndSi_n = paramRndSi[n,:,:]
        for i in np.arange(obsPerInd[n]):
            k += 1
            
            xFix_nt = np.array(xFix[slObsRow[k],:])
            p_nt = p[slObsRow[k]]
            pMat = np.diag(p_nt) - np.outer(p_nt, p_nt)
            hessFix += xFix_nt.T @ pMat @ xFix_nt
            if nRnd:
                xRnd_nt = np.array(xRnd[slObsRow[k],:])
                hessRnd += xRnd_nt.T @ pMat @ xRnd_nt
                
            for l in np.arange(nFix):
                dP_l = dP[slObsRow[k],l]
                dPMat = np.diag(dP_l) - np.outer(dP_l, p_nt) - np.outer(p_nt , dP_l)
                grFixMu[l] += -0.5 * np.trace(paramFixSi @ xFix_nt.T @ dPMat @ xFix_nt)
                if nRnd: grFixMu[l] += -0.5 * np.trace(paramRndSi_n @ xRnd_nt.T @ dPMat @ xRnd_nt)
        
        if nRnd:
            lse -= 0.5 * np.trace(hessRnd @ paramRndSi_n)
            hessRnd = 0
        
    lse -= 0.5 * np.trace(hessFix @ paramFixSi)
    if method[0] == 'qn':
        grFixCh = -0.5 * (hessFix @ paramFixCh + hessFix.T @ paramFixCh)
        grFixCh = np.array(grFixCh[chFixIdx])
    elif method[0] == 'ncvmp':
        grFixSi = -0.5 * hessFix

    return lse, grFixMu, grFixCh, grFixSi  

def elseDeltaRnd(
        method,
        paramFixSi, 
        paramRndMu_n, paramRndCh_n, paramRndSi_n,
        vFix_n, vRnd_n,
        xFix_n, nFix, 
        xRnd_n, nRnd, chRndIdx,
        nObs_n, nRow_n, rowsPerObs_n,
        map_avail_to_obs_n, map_avail_n):
    v = vFix_n + vRnd_n
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300
    nev = map_avail_to_obs_n.T @ ev
    nnev = map_avail_to_obs_n @ nev   
    p = ev / nnev; pVec = p.reshape((nRow_n,1));
    
    lse = -np.sum(np.log(nev))
    pProdX = pVec * xRnd_n
    grRndMu = -np.sum(pProdX, axis = 0)
    grRndCh = 0; grRndSi = 0;
    hessFix = 0; hessRnd = 0;
    dP = pProdX - pVec * (map_avail_n @ pProdX)
    
    for i in np.arange(nObs_n):
        sl = slice(i * rowsPerObs_n[i], (i + 1) * rowsPerObs_n[i])
        xRnd_nt = np.array(xRnd_n[sl,:])
        p_nt = p[sl]
        pMat = np.diag(p_nt) - np.outer(p_nt, p_nt)
        hessRnd += xRnd_nt.T @ pMat @ xRnd_nt
        if nFix: 
            xFix_nt = np.array(xFix_n[sl,:])
            hessFix += xFix_nt.T @ pMat @ xFix_nt
        
        for k in np.arange(nRnd):
            dP_k = dP[sl,k]
            dPMat = np.diag(dP_k) - np.outer(dP_k, p_nt) - np.outer(p_nt, dP_k)
            grRndMu[k] += -0.5 * np.trace(paramRndSi_n @ xRnd_nt.T @ dPMat @ xRnd_nt)
            if nFix: grRndMu[k] += -0.5 * np.trace(paramFixSi @ xFix_nt.T @ dPMat @ xFix_nt)

    lse -= 0.5 * np.trace(hessRnd @ paramRndSi_n)
    if nFix: lse -= 0.5 * np.trace(hessFix @ paramFixSi)
    if method[0] == 'qn':
        grRndCh = -0.5 * (hessRnd @ paramRndCh_n + hessRnd.T @ paramRndCh_n)
        grRndCh = np.array(grRndCh[chRndIdx])
    elif method[0] == 'ncvmp':
        grRndSi = -0.5 * hessRnd.T
            
    return lse, grRndMu, grRndCh, grRndSi   

#QMC
def vFixSim(paramFixMu, paramFixCh, xFix, drawsFix, nDraws, nRow):  
    """
    vFix = np.zeros((nDraws, nRow))
    for d in np.arange(nDraws):
        paramFix = paramFixMu + paramFixCh @ drawsFix[d,:]
        vFix[d,:] = xFix @ paramFix
    """
    paramFix = paramFixMu + (paramFixCh @ drawsFix.T).T
    vFix = (xFix @ paramFix.T).T
    return vFix
 
def vRndSim(
        paramRndMu, paramRndCh, xRnd, drawsRnd, 
        nDraws, nRnd, nInd, nRow, rowsPerInd):
    """
    paramRnd = np.zeros((nInd, nRnd))
    vRnd = np.zeros((nDraws, nRow))
    for d in np.arange(nDraws):
        for n in np.arange(nInd):
            paramRnd[n,:] = paramRndMu[n,:] + paramRndCh[n,:,:] @ drawsRnd[n,d,:]
        paramRndPerRow = np.repeat(paramRnd, rowsPerInd, axis = 0)
        vRnd[d,:] = np.sum(xRnd * paramRndPerRow, axis = 1)
    """
    paramRnd = np.zeros((nInd, nDraws, nRnd))
    for n in np.arange(nInd): 
        paramRnd[n,:,:] = paramRndMu[n,:] + (paramRndCh[n,:,:] @ drawsRnd[n,:,:].T).T
    paramRndPerRow = np.repeat(paramRnd, rowsPerInd, axis = 0)
    vRnd = np.sum(xRnd.reshape((nRow, 1, nRnd)) * paramRndPerRow, axis = 2).T
    return vRnd

def elseQmcFix(
        vFix, vRnd,
        xFix, nFix, chFixIdx,
        drawsFix, nDraws, nRow,
        map_obs_to_ind, map_avail_to_obs):
    lse = 0
    grFixMu = 0
    grFixCh = 0
    
    v = vFix + vRnd
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    for d in np.arange(nDraws):
        nev = map_avail_to_obs.T @ ev[d,:] + 1
        nnev = map_avail_to_obs @ nev   
        pChosen = 1 / nev
        pNonChosen = np.array(ev[d,:] / nnev).reshape((nRow,1))
        lPChosen = np.log(pChosen)
        lse += np.sum(map_obs_to_ind.T @ lPChosen)
        
        derFix = np.ones((nFix,))
        derFixMu = derFix
        derFixCh = derFix[chFixIdx[0]] * drawsFix[d, chFixIdx[1]]
        derFixMuX = derFixMu * xFix
        derFixChX = derFixCh * xFix[:, chFixIdx[0]]
        grFixMu += np.sum(-pNonChosen * derFixMuX, axis = 0)
        grFixCh += np.sum(-pNonChosen * derFixChX, axis = 0)
    
    lse /= nDraws
    grFixMu /= nDraws
    grFixCh /= nDraws
    
    return lse, grFixMu, grFixCh

def elseQmcRnd(
        vFix_n, vRnd_n,
        xRnd_n, nRnd, chRndIdx,
        drawsRnd_n, nDraws, nRow_n,
        map_avail_to_obs_n):
    lse = 0
    grRndMu = 0
    grRndCh = 0
    grRndSi = 0
    
    v = vFix_n + vRnd_n
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    for d in np.arange(nDraws):
        nev = map_avail_to_obs_n.T @ ev[d, :] + 1
        nnev = map_avail_to_obs_n @ nev   
        pChosen = 1 / nev
        pNonChosen = (ev[d, :] / nnev).reshape((nRow_n,1))
        lPChosen = np.log(pChosen)
        lse += np.sum(lPChosen)

        grRndMu += np.sum(-pNonChosen * xRnd_n, axis = 0)
        grRndCh += np.sum(-pNonChosen * xRnd_n[:,chRndIdx[0]] * \
                          drawsRnd_n[d,chRndIdx[1]], axis = 0)                
    
    lse /= nDraws
    grRndMu /= nDraws
    grRndCh /= nDraws
    
    return lse, grRndMu, grRndCh, grRndSi

###
#Updates: Quasi-Newton
###
    
def objectiveQnNextParamFix(
        param, 
        paramRndMu, paramRndCh, paramRndSi,
        mjiAux,
        method,
        vRnd,
        xFix, nFix, chFixIdx, chFixIdxDiag,
        xRnd, nRnd, 
        drawsFix, nDraws, 
        nInd, nRow, obsPerInd, rowsPerInd,
        map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
        slIndObs, slIndRow, slObsInd, slObsRow,
        mu0Fix, Sigma0FixInv):   
    
    paramFixMu = np.array(param[:nFix])
    paramFixCh = np.zeros((nFix, nFix))
    paramFixCh[chFixIdx] = np.array(param[nFix:])
    paramFixChDiag = np.diag(paramFixCh)
    
    ###
    #E-LSE
    ###
    
    vFix = 0
    if method[1] == "qmc":
        vFix = vFixSim(paramFixMu, paramFixCh, xFix, drawsFix, nDraws, nRow)
        lse, grFixMu, grFixCh = elseQmcFix(
                vFix, vRnd,
                xFix, nFix, chFixIdx,
                drawsFix, nDraws, nRow,
                map_obs_to_ind, map_avail_to_obs)
    elif method[1] == "delta":
        vFix = xFix @ paramFixMu
        paramFixSi = paramFixCh @ paramFixCh.T
        lse, grFixMu, grFixCh, _ = elseDeltaFix(
                method,
                paramFixMu, paramFixCh, paramFixSi, 
                paramRndSi,
                vFix, vRnd,
                xFix, nFix, chFixIdx,
                xRnd, nRnd,
                nInd, nRow, obsPerInd,
                map_avail_to_obs, map_avail,
                slObsInd, slObsRow)
    elif method[1] == "mji":      
        paramFixSi = paramFixCh @ paramFixCh.T
        lse, grFixMu, grFixCh, _ = elseMjiFix(
                method,
                paramFixMu, paramFixCh, paramFixSi,
                paramRndMu, paramRndCh, paramRndSi,
                mjiAux,
                xFix, nFix, chFixIdx,
                xRnd, nRnd,
                nInd, nRow, rowsPerInd,
                map_avail_to_obs, slIndObs, slIndRow)
    
    ###
    #Prior
    ###
    
    lPrior = \
    -0.5 * np.trace(paramFixCh @ paramFixCh.T @ Sigma0FixInv) \
    -0.5 * paramFixMu @ Sigma0FixInv @ paramFixMu \
    + paramFixMu @ Sigma0FixInv @ mu0Fix \
    + np.sum(np.log(np.absolute(paramFixChDiag)))
    
    grFixMuPrior = -Sigma0FixInv @ paramFixMu + Sigma0FixInv @ mu0Fix
    grFixChPriorAux = -Sigma0FixInv @ paramFixCh
    grFixChPrior = np.array(grFixChPriorAux[chFixIdx])
    grFixChPrior[chFixIdxDiag] += 1 / paramFixChDiag
    
    ###
    #E-LSE + prior
    ###    
    
    ll = -(lse + lPrior)
    
    grFixMu += grFixMuPrior
    grFixCh += grFixChPrior
    gr = -np.concatenate((grFixMu, grFixCh))
    
    return ll, gr

def nextQnParamFix(
        method,
        paramFixMu, paramFixCh,
        paramRndMu, paramRndCh, paramRndSi,
        mjiAux,
        xFix, nFix, chFixIdx, chFixIdxDiag,
        xRnd, nRnd, 
        drawsFix, drawsRnd, nDraws, 
        nInd, nObs, nRow, 
        obsPerInd, rowsPerInd,
        map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
        slIndObs, slIndRow, slObsInd, slObsRow,        
        mu0Fix, Sigma0FixInv):      
    
    vRnd = 0 
    if method[1] == "qmc" and nRnd: 
        vRnd = vRndSim(
                paramRndMu, paramRndCh, xRnd, drawsRnd, 
                nDraws, nRnd, nInd, nRow, rowsPerInd)
    elif method[1] == "delta" and nRnd:
        paramRndPerRow = np.repeat(paramRndMu, rowsPerInd, axis = 0)
        vRnd = np.sum(xRnd * paramRndPerRow, axis = 1)
    
    inits = np.concatenate((paramFixMu, paramFixCh[chFixIdx]))    
    resOpt = sp.optimize.minimize(
            fun = objectiveQnNextParamFix,
            x0 = inits,
            args = (paramRndMu, paramRndCh, paramRndSi,
                    mjiAux,
                    method,
                    vRnd,
                    xFix, nFix, chFixIdx, chFixIdxDiag,
                    xRnd, nRnd, 
                    drawsFix, nDraws, 
                    nInd, nRow, obsPerInd, rowsPerInd,
                    map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
                    slIndObs, slIndRow, slObsInd, slObsRow,
                    mu0Fix, Sigma0FixInv),
            method = 'BFGS',
            jac = True,
            options = {'disp': False})
    paramFixMu = np.array(resOpt['x'][:nFix])
    paramFixCh[chFixIdx] = np.array(resOpt['x'][nFix:])
    paramFixSi = paramFixCh @ paramFixCh.T
    
    return paramFixMu, paramFixCh, paramFixSi           
        
def objectiveQnNextParamRnd(
        param,
        paramFixMu, paramFixSi, 
        a_n,
        method,
        vFix_n,
        xFix_n, nFix,
        xRnd_n, nRnd, chRndIdx, chRndIdxDiag,
        drawsRnd_n, nDraws, 
        nObs_n, nRow_n, rowsPerObs_n,
        map_obs_to_ind_n, map_avail_to_obs_n, map_ind_to_avail_n, map_avail_n,
        zetaMu, psiInv, omega):

    paramRndMu_n = np.array(param[:nRnd])
    paramRndCh_n = np.zeros((nRnd, nRnd))
    paramRndCh_n[chRndIdx] = np.array(param[nRnd:])
    paramRndChDiag_n = np.diag(paramRndCh_n)  
    paramRndSi_n = paramRndCh_n @ paramRndCh_n.T
    
    ###
    #E-LSE
    ###
    
    if method[1] == "qmc":
        vRnd_n = np.zeros((nDraws, nRow_n))
        for d in np.arange(nDraws):
            paramRnd = paramRndMu_n + paramRndCh_n @ drawsRnd_n[d,:]
            vRnd_n[d, :] = xRnd_n @ paramRnd
        lse, grRndMu, grRndCh, _ = elseQmcRnd(
                vFix_n, vRnd_n,
                xRnd_n, nRnd, chRndIdx,
                drawsRnd_n, nDraws, nRow_n,
                map_avail_to_obs_n)
    elif method[1] == "delta":
        vRnd_n = xRnd_n @ paramRndMu_n
        lse, grRndMu, grRndCh, _ = elseDeltaRnd(
                method,
                paramFixSi, 
                paramRndMu_n, paramRndCh_n, paramRndSi_n,
                vFix_n, vRnd_n,
                xFix_n, nFix, 
                xRnd_n, nRnd, chRndIdx,
                nObs_n, nRow_n, rowsPerObs_n,
                map_avail_to_obs_n, map_avail_n)
    elif method[1] == "mji":
        lse, grRndMu, grRndCh, _ = elseMjiRnd(
                method,
                paramFixMu, paramFixSi,
                paramRndMu_n, paramRndCh_n, paramRndSi_n,
                a_n,
                xFix_n, nFix, 
                xRnd_n, nRnd, chRndIdx,
                nRow_n,
                map_avail_to_obs_n)
            
    ###
    #Prior
    ###
    
    lPrior = \
    -(omega / 2) * np.trace(paramRndCh_n @ paramRndCh_n.T @ psiInv) \
    -(omega / 2) * paramRndMu_n @ psiInv @ paramRndMu_n \
    + omega * paramRndMu_n @ psiInv @ zetaMu \
    + np.sum(np.log(np.absolute(paramRndChDiag_n)))

    grRndMuPrior = -omega * psiInv @ paramRndMu_n + omega * psiInv @ zetaMu
    grRndChPriorAux = -omega * psiInv @ paramRndCh_n
    grRndChPrior = np.array(grRndChPriorAux[chRndIdx])
    grRndChPrior[chRndIdxDiag] += 1 / paramRndChDiag_n
    
    ###
    #E-LSE + prior
    ###    
    
    ll = -(lse + lPrior)
    
    grRndMu += grRndMuPrior
    grRndCh += grRndChPrior
    gr = -np.concatenate((grRndMu, grRndCh))
    return ll, gr

def nextQnParamRnd(
        method,
        paramFixMu, paramFixCh, paramFixSi,
        paramRndMu, paramRndCh, paramRndSi,
        mjiAux,
        xFix, nFix, 
        xRnd, nRnd, chRndIdx, chRndIdxDiag,
        drawsFix, drawsRnd, nDraws, nInd, nRow, 
        obsPerInd, rowsPerInd, rowsPerObs,
        map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
        slIndObs, slIndRow, slObsRow,
        zetaMu, psiInv, omega): 
    
    vFix = 0;
    if method[1] == "qmc" and nFix: 
        vFix = vFixSim(paramFixMu, paramFixCh, xFix, drawsFix, nDraws, nRow)
    elif method[1] == "delta" and nFix:
        vFix = xFix @ paramFixMu
    
    a_n = None
    vFix_n = 0
    xFix_n = xFix
    xRnd_n = 0
    drawsRnd_n = None
    map_obs_to_ind_n = None
    map_ind_to_avail_n = None
    map_ind_to_avail_n = None
    map_avail_n = None
    rowsPerObs_n = None
        
    for n in np.arange(nInd):
        xRnd_n = xRnd[slIndRow[n],:]
        map_avail_to_obs_n = map_avail_to_obs[slIndRow[n], slIndObs[n]]
        nObs_n = obsPerInd[n]
        nRow_n = rowsPerInd[n]
        if method[1] == "qmc":  
            if nFix: vFix_n = vFix[:,slIndRow[n]]
            drawsRnd_n = drawsRnd[n,:,:]
            map_obs_to_ind_n = map_obs_to_ind[slIndObs[n], n]
            map_avail_to_obs_n = map_avail_to_obs[slIndRow[n], slIndObs[n]]
            map_ind_to_avail_n = map_ind_to_avail[n, slIndRow[n]]
        elif method[1] == "delta":
            if nFix: 
                xFix_n = xFix[slIndRow[n],:]
                vFix_n = vFix[slIndRow[n]]
            nObs_n = obsPerInd[n]
            rowsPerObs_n = rowsPerObs[slIndObs[n]]
            map_avail_n = map_avail[slIndRow[n], slIndRow[n]]
        elif method[1] == "mji":
            if nFix: xFix_n = xFix[slIndRow[n],:]
            a_n = mjiAux[slIndRow[n]]
            
        inits = np.concatenate((paramRndMu[n,:], paramRndCh[n,chRndIdx[0],chRndIdx[1]]))
        resOpt = sp.optimize.minimize(
                fun = objectiveQnNextParamRnd,
                x0 = inits,
                args = (paramFixMu, paramFixSi,
                        a_n,
                        method,
                        vFix_n,
                        xFix_n, nFix,
                        xRnd_n, nRnd, chRndIdx, chRndIdxDiag,
                        drawsRnd_n, nDraws, 
                        nObs_n, nRow_n, rowsPerObs_n,
                        map_obs_to_ind_n, map_avail_to_obs_n, map_ind_to_avail_n, map_avail_n,
                        zetaMu, psiInv, omega),
                        method = 'BFGS',
                        jac = True,
                        options = {'disp': False})
        paramRndMu[n,:] = np.array(resOpt['x'][:nRnd])
        paramRndCh_n = np.zeros((nRnd, nRnd))
        paramRndCh_n[chRndIdx] = np.array(resOpt['x'][nRnd:])
        paramRndCh[n,:,:] = np.array(paramRndCh_n)
        paramRndSi[n,:,:] = paramRndCh_n @ paramRndCh_n.T
        
        #print(n)
    
    return paramRndMu, paramRndCh, paramRndSi

###
#Updates: NCVMP
##
    
def ncvmpUpdate(grMuElbo, grSigmaElbo, mu):
    Sigma = -np.linalg.inv(2 * grSigmaElbo)
    muNew = mu + Sigma @ grMuElbo
    return muNew, Sigma
    
def nextNcvmpParamFix(
        method,
        paramFixMu, paramFixCh, paramFixSi,
        paramRndMu, paramRndCh, paramRndSi,
        mjiAux,
        xFix, nFix, chFixIdx,
        xRnd, nRnd, 
        nInd, nObs, nRow, 
        obsPerInd, rowsPerInd,
        map_avail_to_obs, map_avail,
        slIndObs, slIndRow, slObsInd, slObsRow,
        mu0Fix, Sigma0FixInv):
    
    ###
    #E-LSE
    ###
    
    vRnd = 0 
    if method[1] == "delta" and nRnd:
        paramRndPerRow = np.repeat(paramRndMu, rowsPerInd, axis = 0)
        vRnd = np.sum(xRnd * paramRndPerRow, axis = 1)    
    
    vFix = 0
    if method[1] == "delta":
        vFix = xFix @ paramFixMu
        _, grFixMu, _, grFixSi = elseDeltaFix(
                method,
                paramFixMu, paramFixCh, paramFixSi, 
                paramRndSi,
                vFix, vRnd,
                xFix, nFix, chFixIdx,
                xRnd, nRnd,
                nInd, nRow, obsPerInd,
                map_avail_to_obs, map_avail,
                slObsInd, slObsRow)
    elif method[1] == "mji":      
        _, grFixMu, _, grFixSi = elseMjiFix(
                method,
                paramFixMu, paramFixCh, paramFixSi,
                paramRndMu, paramRndCh, paramRndSi,
                mjiAux,
                xFix, nFix, chFixIdx,
                xRnd, nRnd,
                nInd, nRow, rowsPerInd,
                map_avail_to_obs, slIndObs, slIndRow)
    
    ###
    #Prior
    ###
    
    grFixMuPrior = -Sigma0FixInv @ paramFixMu + Sigma0FixInv @ mu0Fix
    grFixSiPrior = -0.5 * Sigma0FixInv

    ###
    #E-LSE + prior
    ###    
    
    grFixMu += grFixMuPrior
    grFixSi += grFixSiPrior
    
    ###
    #Update
    ###    
    
    paramFixMu, paramFixSi = ncvmpUpdate(grFixMu, grFixSi, paramFixMu)
    paramFixCh = np.linalg.cholesky(paramFixSi)
    
    return paramFixMu, paramFixCh, paramFixSi

def nextNcvmpParamRnd(
        method,
        paramFixMu, paramFixSi,
        paramRndMu, paramRndCh, paramRndSi,
        mjiAux,
        xFix, nFix, 
        xRnd, nRnd, chRndIdx, chRndIdxDiag,
        nInd, 
        obsPerInd, rowsPerInd, rowsPerObs,
        map_avail_to_obs, map_avail,
        slIndObs, slIndRow,
        zetaMu, psiInv, omega):
    
    vFix = 0
    if method[1] == "delta" and nFix:
        vFix = xFix @ paramFixMu
    
    a_n = None
    vFix_n = 0
    xFix_n = xFix
    xRnd_n = 0
    rowsPerObs_n = None
        
    for n in np.arange(nInd):
        paramRndMu_n = paramRndMu[n,:]
        paramRndCh_n = paramRndCh[n,:,:]
        paramRndSi_n = paramRndSi[n,:,:]
        
        xRnd_n = xRnd[slIndRow[n],:]
        map_avail_to_obs_n = map_avail_to_obs[slIndRow[n], slIndObs[n]]
        map_avail_n = map_avail[slIndRow[n], slIndRow[n]]
        nObs_n = obsPerInd[n]
        nRow_n = rowsPerInd[n]
        if method[1] == "delta":
            if nFix: 
                xFix_n = xFix[slIndRow[n],:]
                vFix_n = vFix[slIndRow[n]]
            nObs_n = obsPerInd[n]
            rowsPerObs_n = rowsPerObs[slIndObs[n]]
        elif method[1] == "mji":
            if nFix: xFix_n = xFix[slIndRow[n],:]
            a_n = mjiAux[slIndRow[n]]    
    
        ###
        #E-LSE
        ###
        
        if method[1] == "delta":
            vRnd_n = xRnd_n @ paramRndMu_n
            _, grRndMu, _, grRndSi = elseDeltaRnd(
                    method,
                    paramFixSi, 
                    paramRndMu_n, paramRndCh_n, paramRndSi_n,
                    vFix_n, vRnd_n,
                    xFix_n, nFix, 
                    xRnd_n, nRnd, chRndIdx,
                    nObs_n, nRow_n, rowsPerObs_n,
                    map_avail_to_obs_n, map_avail_n)
        elif method[1] == "mji":
            _, grRndMu, _, grRndSi = elseMjiRnd(
                    method,
                    paramFixMu, paramFixSi,
                    paramRndMu_n, paramRndCh_n, paramRndSi_n,
                    a_n,
                    xFix_n, nFix, 
                    xRnd_n, nRnd, chRndIdx,
                    nRow_n,
                    map_avail_to_obs_n)
        
        ###
        #Prior
        ###
        
        grRndMuPrior = -omega * psiInv @ paramRndMu_n + omega * psiInv @ zetaMu
        grRndSiPrior = -(omega / 2) * psiInv
    
        ###
        #E-LSE + prior
        ###    
        
        grRndMu += grRndMuPrior
        grRndSi += grRndSiPrior
        
        ###
        #Update
        ###    
        
        paramRndMu_n, paramRndSi_n = ncvmpUpdate(grRndMu, grRndSi, paramRndMu_n)
        paramRndMu[n,:] = paramRndMu_n
        paramRndCh[n,:,:] = np.linalg.cholesky(paramRndSi_n)
        paramRndSi[n,:,:] = paramRndSi_n
    
    return paramRndMu, paramRndCh, paramRndSi

###
#Coordinate ascent
###

def coordinateAscent(
        method, vb_iter, vb_tol,
        paramFixMu, paramFixCh, paramFixSi,
        paramRndMu, paramRndCh, paramRndSi,
        mjiAux,
        xFix, nFix, chFixIdx, chFixIdxDiag, 
        xRnd, nRnd, chRndIdx, chRndIdxDiag,
        drawsFix, drawsRnd, nDraws, 
        nInd, nObs, nRow,
        obsPerInd, rowsPerInd, rowsPerObs,
        map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
        slIndObs, slIndRow, slObsInd, slObsRow,
        mu0Fix, Sigma0FixInv,
        mu0Rnd, Sigma0RndInv,
        zetaMu, zetaSi,
        psi, psiInv, omega, nu,
        cK, dK, rK):
    
    ###
    #Initialise
    ###
    
    iters = 0
    parOld = np.concatenate((paramFixMu, np.diag(paramFixSi),
                             zetaMu, np.diag(psi), dK))
    parmat = np.zeros((5, parOld.shape[0]))
    parChange = 1e200
    parChangeOld = parChange
    parChangeDiff = 0
    
    ###
    #CAVI
    ###
    
    while iters < vb_iter and parChange >= vb_tol and parChangeDiff < 0.1:
        iters += 1
        
        if nRnd:
            #beta
            if method[0] == "qn":
                paramRndMu, paramRndCh, paramRndSi = nextQnParamRnd(
                        method,
                        paramFixMu, paramFixCh, paramFixSi,
                        paramRndMu, paramRndCh, paramRndSi,
                        mjiAux,
                        xFix, nFix, 
                        xRnd, nRnd, chRndIdx, chRndIdxDiag,
                        drawsFix, drawsRnd, nDraws, nInd, nRow, 
                        obsPerInd, rowsPerInd, rowsPerObs,
                        map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
                        slIndObs, slIndRow, slObsRow,
                        zetaMu, psiInv, omega)
            elif method[0] == "ncvmp":
                paramRndMu, paramRndCh, paramRndSi = nextNcvmpParamRnd(
                        method,
                        paramFixMu, paramFixSi,
                        paramRndMu, paramRndCh, paramRndSi,
                        mjiAux,
                        xFix, nFix, 
                        xRnd, nRnd, chRndIdx, chRndIdxDiag,
                        nInd, 
                        obsPerInd, rowsPerInd, rowsPerObs,
                        map_avail_to_obs, map_avail,
                        slIndObs, slIndRow,
                        zetaMu, psiInv, omega)
            
            #zeta
            zetaSi = np.linalg.inv(Sigma0RndInv + nInd * omega * psiInv)
            zetaMu = zetaSi @ (Sigma0RndInv @ mu0Rnd +\
                               omega * psiInv @ np.sum(paramRndMu, axis = 0))

            #Omega
            betaS = paramRndMu - zetaMu
            psi = 2 * nu * np.diag(cK / dK) + nInd * zetaSi +\
            np.sum(paramRndSi, axis = 0) + betaS.T @ betaS
            psiInv = np.linalg.inv(psi)
            
            #iwishDiagA
            dK = rK + nu * omega * np.diag(psiInv)
            
        if nFix:
            #alpha
            if method[0] == "qn":
                paramFixMu, paramFixCh, paramFixSi = nextQnParamFix(
                        method,
                        paramFixMu, paramFixCh,
                        paramRndMu, paramRndCh, paramRndSi,
                        mjiAux,
                        xFix, nFix, chFixIdx, chFixIdxDiag,
                        xRnd, nRnd, 
                        drawsFix, drawsRnd, nDraws, 
                        nInd, nObs, nRow, 
                        obsPerInd, rowsPerInd,
                        map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
                        slIndObs, slIndRow, slObsInd, slObsRow,        
                        mu0Fix, Sigma0FixInv)    

            elif method[0] == "ncvmp":
                paramFixMu, paramFixCh, paramFixSi = nextNcvmpParamFix(
                        method,
                        paramFixMu, paramFixCh, paramFixSi,
                        paramRndMu, paramRndCh, paramRndSi,
                        mjiAux,
                        xFix, nFix, chFixIdx,
                        xRnd, nRnd, 
                        nInd, nObs, nRow, 
                        obsPerInd, rowsPerInd,
                        map_avail_to_obs, map_avail,
                        slIndObs, slIndRow, slObsInd, slObsRow, 
                        mu0Fix, Sigma0FixInv)
        
        #mjiAux
        if method[1] == "mji":
            mjiAux = nextMjiAux(
                    paramFixMu, paramFixSi,
                    paramRndMu, paramRndSi,
                    mjiAux,
                    xFix, nFix,
                    xRnd, nRnd,
                    nInd, nRow,
                    map_avail, slIndRow)
        
        ###
        #Check for convergence
        ###
        
        par = np.concatenate((paramFixMu, np.diag(paramFixSi),
                              zetaMu, np.diag(psi), dK)).reshape((1,-1))
        parmat = np.vstack((parmat[1:,:], par))
        parNew = np.mean(parmat, axis = 0)
        if iters > 5:
            parChange = np.max(np.absolute((parNew - parOld)) / 
                               np.absolute(parOld + 1e-8))
            parChangeDiff = parChange - parChangeOld
            parChangeOld = parChange
        parOld = parNew
          
        ###
        #Display progress
        ###
        
        print(" ")
        print('Iteration ' + str(iters) + 
              '; max. rel. change param.: ' + str(parChange) + ';')
        #print(par)
        
        if nFix:
            print("paramFixMu:")
            print(paramFixMu)
            print("diag(paramFixSi):")
            print(np.diag(paramFixSi))
        if nRnd:
            print("zetaMu:")
            print(zetaMu)
            print("diag(psi):")
            print(np.diag(psi))
            print("dK:")
            print(dK)        
        
    return (paramFixMu, paramFixCh, paramFixSi,
            paramRndMu, paramRndCh, paramRndSi,
            zetaMu, zetaSi, psi, dK,
            mjiAux,
            iters, parChange)

###
#Estimate
###

def estimate(
        method, vb_iter, vb_tol, modelName, seed,
        drawsType, nDraws, deleteDraws,
        paramFixMu_inits, paramFixSi_inits,
        paramRndMu_inits, paramRndSi_inits,
        zetaMu_inits, zetaSi_inits,
        cK, dK_inits, rK,
        omega, psi_inits, nu, 
        mu0Fix, Sigma0FixInv,
        mu0Rnd, Sigma0RndInv,
        indID, obsID, chosen,
        xFix, xRnd):
    
    np.random.seed(seed)
    
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    
    full = method[1] in ['delta','mji']
    xList = [xFix, xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen, full)
    xFix, xRnd = xList[0], xList[1] 
    
    nRow = map_avail_to_obs.shape[0]
    obsPerInd = np.ones((nObs,), dtype = 'int64') @ map_obs_to_ind
    map_ind_to_avail = (map_avail_to_obs @ map_obs_to_ind).T
    map_avail = map_avail_to_obs @ map_avail_to_obs.T
    slObsInd = map_obs_to_ind @ np.arange(nInd)
    
    cumRowsPerObs = np.cumsum(rowsPerObs)
    slObsRow = [slice(l,u) for l,u in zip(np.concatenate((np.array([0]), cumRowsPerObs[:-1])),
                      cumRowsPerObs)]
    
    chFixIdx = None; chFixIdxDiag = None;
    if nFix: 
        chFixIdx = np.triu_indices(nFix); chFixIdx = chFixIdx[1], chFixIdx[0];
        chFixIdxDiagAux = np.ones((nFix, nFix), dtype = 'int64')
        chFixIdxDiagAux[chFixIdx] = np.arange((nFix * (nFix + 1) / 2))
        chFixIdxDiag = np.diag(chFixIdxDiagAux)
    
    chRndIdx = None; chRndIdxDiag = None;
    if nRnd: 
        chRndIdx = np.triu_indices(nRnd); chRndIdx = chRndIdx[1], chRndIdx[0];
        chRndIdxDiagAux = np.ones((nRnd, nRnd), dtype = 'int64')
        chRndIdxDiagAux[chRndIdx] = np.arange((nRnd * (nRnd + 1) / 2))
        chRndIdxDiag = np.diag(chRndIdxDiagAux)
    
    slIndObs = None; slIndRow = None;
    cumObsPerInd = np.cumsum(obsPerInd)
    slIndObs = [slice(l,u) for l,u in zip(np.concatenate((np.array([0]), cumObsPerInd[:-1])), 
                      cumObsPerInd)]
    cumRowsPerInd = np.cumsum(rowsPerInd)
    slIndRow = [slice(l,u) for l,u in zip(np.concatenate((np.array([0]), cumRowsPerInd[:-1])), 
                      cumRowsPerInd)]
           
    ### 
    #Generate draws
    ###
    
    drawsFix = None; drawsRnd = None;
    if method[1] == "qmc": 
        if nFix: _, drawsFix = makeNormalDraws(nDraws, nFix, drawsType)
        if nRnd: drawsRnd, _ = makeNormalDraws(nDraws, nRnd, drawsType, nInd)     
    
    ### 
    #Coordinate Ascent
    ###
    
    if method[0] == "qn" and (method[1] in ["qmc","delta"]) and nRnd == 0: vb_iter = 1
    
    paramFixMu = np.zeros((0,)); paramFixCh = np.zeros((0, 0)); paramFixSi = np.zeros((0, 0));  
    if nFix:
        paramFixMu = paramFixMu_inits.copy()
        paramFixCh = np.linalg.cholesky(paramFixSi_inits).copy()
        paramFixSi = paramFixSi_inits.copy()
    
    paramRndMu = None; paramRndCh = None; paramRndSi = None;
    zetaMu = np.zeros((0,)); zetaSi = None;
    dK = np.zeros((0,));
    psi = np.zeros((0, 0)); psiInv = None;
    if nRnd:
        paramRndMu = paramRndMu_inits.copy()
        paramRndCh = np.linalg.cholesky(paramRndSi_inits).copy()
        paramRndSi = paramRndSi_inits.copy()
        zetaMu = zetaMu_inits.copy()
        zetaSi = zetaSi_inits.copy()
        dK = dK_inits.copy()
        psi = psi_inits.copy()
        psiInv = np.linalg.inv(psi).copy()
     
    mjiAux = None
    if method[1] == "mji":
        numer = np.ones((nRow,))
        denom = map_avail @ numer
        mjiAux = np.array(numer / denom).reshape((nRow, 1))
    
    tic = time.time()
    (paramFixMu, paramFixCh, paramFixSi,
     paramRndMu, paramRndCh, paramRndSi,
     zetaMu, zetaSi, psi, dK,
     mjiAux,
     iters, parChange) = coordinateAscent(
             method, vb_iter, vb_tol,
             paramFixMu, paramFixCh, paramFixSi,
             paramRndMu, paramRndCh, paramRndSi,
             mjiAux,
             xFix, nFix, chFixIdx, chFixIdxDiag, 
             xRnd, nRnd, chRndIdx, chRndIdxDiag,
             drawsFix, drawsRnd, nDraws, 
             nInd, nObs, nRow,
             obsPerInd, rowsPerInd, rowsPerObs,
             map_obs_to_ind, map_avail_to_obs, map_ind_to_avail, map_avail,
             slIndObs, slIndRow, slObsInd, slObsRow,
             mu0Fix, Sigma0FixInv,
             mu0Rnd, Sigma0RndInv,
             zetaMu, zetaSi,
             psi, psiInv, omega, nu,
             cK, dK, rK)
    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))

    ###
    #Delete draws
    ###      
    
    if deleteDraws:
        drawsFix = None; drawsRnd = None; 
    
    ###
    #Save results
    ###
    
    results = {'modelName': modelName, 'seed': seed,
               'estimation_time': toc, 'iters': iters, 'termTol': parChange,
               'drawsType': drawsType, 'nDraws': nDraws,
               'drawsFix': drawsFix, 'drawsRnd': drawsRnd,
               'paramFixMu': paramFixMu, 'paramFixCh': paramFixCh, 'paramFixSi': paramFixSi,
               'paramRndMu': paramRndMu, 'paramRndCh': paramRndCh, 'paramRndSi': paramRndSi,
               'zetaMu': zetaMu, 'zetaSi': zetaSi, 'psi': psi, 'omega': omega, 'dK': dK,
               'mjiAux': mjiAux
               }
        
    return results

def inits(indID, xFix, xRnd, nu, A):
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    
    nInd = np.unique(indID).shape[0]
    
    paramFixMu_inits = None; paramFixSi_inits = None;
    if nFix:
        paramFixMu_inits = np.zeros((nFix,))
        paramFixSi_inits = 0.1 * np.eye(nFix)
        
    paramRndMu_inits = None; paramRndSi_inits = None;
    zetaMu_inits = None; zetaSi_inits = None;
    cK = None; dK_inits = None; rK = None
    omega = None; psi_inits = None
    if nRnd:
        paramRndMu_inits = np.zeros((nInd, nRnd))
        paramRndSi_inits = np.repeat(0.1 * np.eye(nRnd).reshape(1, nRnd, nRnd), 
                                 nInd, axis = 0)
        zetaMu_inits = np.zeros((nRnd,))
        zetaSi_inits = 0.1 * np.eye(nRnd)
        cK = (nu + nRnd) / 2.0
        dK_inits = cK * np.ones((nRnd,))
        rK = A**(-2) * np.ones((nRnd,))
        omega = nu + nInd + nRnd - 1
        psi_inits = (omega - nRnd + 1) * np.eye(nRnd)
    
    return (paramFixMu_inits, paramFixSi_inits,
            paramRndMu_inits, paramRndSi_inits,
            zetaMu_inits, zetaSi_inits,
            cK, dK_inits, rK,
            omega, psi_inits)

###
#Prediction
###
    
def predict(
        nIter, nTakes, nSim, seed,
        paramFixMu, paramFixCh,
        zetaMu, zetaSi, psi, omega,
        indID, obsID, altID, chosen,
        xFix, xRnd):
    
    np.random.seed(seed)
    
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
    
    chIdx = np.triu_indices(nRnd); chIdx = chIdx[1], chIdx[0];

    ###
    #Prediction
    ###
    
    pPred = np.zeros((nRow + nObs,))  
    vFix = 0 
    
    zetaCh = np.linalg.cholesky(zetaSi)
    
    for i in np.arange(nIter):
        if nFix: 
            paramFix = paramFixMu + paramFixCh @ np.random.randn(nFix,)
            vFix = np.tile(xFix @ paramFix, (nSim,))
        zeta = zetaMu + zetaCh @ np.random.randn(nRnd,)
        ch = np.linalg.cholesky(invwishart.rvs(omega, psi).reshape((nRnd, nRnd)))
        
        pPred_iter = np.zeros((nRow + nObs,))
        
        for t in np.arange(nTakes):
            paramRnd = zeta + (ch @ np.random.randn(nRnd, nInd * nSim)).T
            paramRndPerRow = np.repeat(paramRnd, sim_rowsPerInd, axis = 0)
            vRnd = np.sum(sim_xRnd * paramRndPerRow, axis = 1)
            
            pPred_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
            pPred_iter += pPred_take 
        pPred += (pPred_iter / nTakes)
    pPred /= nIter  
    return pPred

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
    #data = data[data['ID'] <= 100]
    
    ###
    #Prepare data
    ###
    
    indID = np.array(data['ID'].values, dtype = 'int64').reshape((-1,1))
    obsID = np.array(data['obsID'].values, dtype = 'int64').reshape((-1,1))
    altID = np.array(data['altID'].values, dtype = 'int64').reshape((-1,1))
    
    chosen = np.array(data['chosen'].values, dtype = 'int64')
    
    tt = np.array(data['TT'].values, dtype = 'float64').reshape((-1,1)) / 10
    cost = np.array(data['CO'].values, dtype = 'float64').reshape((-1,1)) / 10
    he = np.array(data['HE'].values, dtype = 'float64').reshape((-1,1)) / 10
    ga = np.array(data['GA'].values, dtype = 'int64').reshape((-1,1))
    cost[(altID[:,0] <= 2) & (ga[:,0] == 1),0] = 0
    
    const2 = 1 * (altID == 2)
    const3 = 1 * (altID == 3)
    """
    ###
    #Generate data
    ###
    
    N = 500
    T = 5
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
    
    xFix = np.random.rand(NTJ, L)
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
    #Estimate MXL via VB
    ###
    
    #xFix = np.zeros((0,0)) #np.hstack((const2, const3, cost, tt))
    #xRnd = np.array(xRnd[:,:5]) #np.zeros((0,0))
    
    mu0Fix = np.zeros((xFix.shape[1],))
    Sigma0FixInv = 1e-6 * np.eye(xFix.shape[1])
    mu0Rnd = np.zeros((xRnd.shape[1],))
    Sigma0RndInv = 1e-6 * np.eye(xRnd.shape[1]) 
    nu = 2; A = 1.04;
    
    (paramFixMu_inits, paramFixSi_inits,
     paramRndMu_inits, paramRndSi_inits,
     zetaMu_inits, zetaSi_inits,
     cK, dK_inits, rK,
     omega, psi_inits) = inits(indID, xFix, xRnd, nu, A)
    
    method = ('qn', 'qmc')
    vb_iter = 500
    vb_tol = 0.005
    
    drawsType = 'mlhs'
    nDraws = 100
    seed = 4711

    modelName = 'test'
    deleteDraws = True
    
    results = estimate(
            method, vb_iter, vb_tol, modelName, seed,
            drawsType, nDraws, deleteDraws,
            paramFixMu_inits, paramFixSi_inits,
            paramRndMu_inits, paramRndSi_inits,
            zetaMu_inits, zetaSi_inits,
            cK, dK_inits, rK,
            omega, psi_inits, nu, 
            mu0Fix, Sigma0FixInv,
            mu0Rnd, Sigma0RndInv,
            indID, obsID, chosen,
            xFix, xRnd)    