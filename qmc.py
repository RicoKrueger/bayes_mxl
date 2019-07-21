import numpy as np
import scipy.stats

###
#Define QMC methods
###

#Pseudo-random
def pseudoRandom(nSeq, nDim):
    seq = np.random.rand(nSeq, nDim)
    return seq

#Modified Latin Hypercube sampling
def mlhs(nSeq, nDim):
    seq = np.empty((nSeq, nDim))
    h = np.arange(nSeq)
    for i in np.arange(nDim):
        d = h + np.random.rand()
        seq[:, i] = d[np.random.choice(nSeq, size = nSeq, replace = False)]
    seq /= nSeq
    return seq

#Halton
primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997])

def vdc(nSeq, base = 2):
    seq = np.zeros((nSeq,))
    for i in np.arange(nSeq):
        q = i + 1
        denom = 1
        while q > 0:
            denom *= base
            q, mod = np.divmod(q, base)
            seq[i] += mod / float(denom)
    return seq

def halton(nSeq, nDim):
    seq = np.empty((nSeq, nDim))
    for i in np.arange(nDim):
        seq[:, i] = vdc(nSeq, primes[i])
    return seq

def haltonShiftShuffle(nSeq, nDim):
    seq = halton(nSeq, nDim)
    for i in np.arange(nDim):
        seq[:, i] += np.random.rand()
        seq[:, i] -= np.floor(seq[:, i])
        seq[:, i] = seq[np.random.choice(nSeq, size = nSeq, replace = False), i]
    return seq

#Extensible shifted lattice points 
"""
def eslp(nSeq, nDim, base = 2, eta = 1571):
    h = np.array(eta ** np.arange(nDim)).reshape((1, nDim))
    vdcSeq = np.array(vdc(nSeq, base)).reshape((nSeq, 1))
    u = np.random.rand(1,nDim)
    seq = h * vdcSeq + u
    seq -= np.floor(seq)
    seq = 1 - np.absolute(2 * seq - 1)
    return seq
"""

###
#Make draws
###   
    
def makeUniformDraws(nSeq, nDim, drawsType, nInd = 1):
    drawsArr = np.zeros((nInd, nSeq, nDim))
    for n in np.arange(nInd):
        if drawsType == "pseudoRandom":
            drawsArr[n,:,:] = pseudoRandom(nSeq, nDim)
        elif drawsType == "mlhs":
            drawsArr[n,:,:] = mlhs(nSeq, nDim)
        elif drawsType == "halton":
            drawsArr[n,:,:] = haltonShiftShuffle(nSeq, nDim)
        elif drawsType == "haltonShiftShuffle":
            drawsArr[n,:,:] = haltonShiftShuffle(nSeq, nDim)
        else:
            assert False, "drawsType unknown!"
    drawsMat = np.moveaxis(drawsArr, [0, 1], [1, 0]).reshape((nInd * nSeq, nDim))
    return drawsArr, drawsMat

def makeNormalDraws(nSeq, nDim, drawsType, nInd = 1):
    drawsArr, _ = makeUniformDraws(nSeq, nDim, drawsType, nInd)            
    drawsArr = np.array(scipy.stats.norm.ppf(drawsArr))
    drawsMat = np.moveaxis(drawsArr, [0, 1], [1, 0]).reshape((nInd * nSeq, nDim))
    return drawsArr, drawsMat
            
###
#If main: Test methods
###
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    nSeq = 64
    nDim = 2
    
    seq_pseudoRandom = pseudoRandom(nSeq, nDim)
    seq_mlhs = mlhs(nSeq, nDim)
    seq_halton = halton(nSeq, nDim)
    seq_haltonShiftShuffle = haltonShiftShuffle(nSeq, nDim)
    
    fig, ax = plt.subplots()
    ax.scatter(seq_pseudoRandom[:, 0], 
               seq_pseudoRandom[:, 1], label = "Pseudo-random")
    ax.scatter(seq_mlhs[:, 0], 
               seq_mlhs[:, 1], label = "MLHS")
    ax.scatter(seq_halton[:, 0], 
               seq_halton[:, 1], label = "Halton")
    ax.scatter(seq_haltonShiftShuffle[:, 0], 
               seq_haltonShiftShuffle[:, 1], label = "Halton (shifted, shuffled)")
    
    ax.grid()
    plt.legend()
    plt.show()
    
    nSeq, nDim, nInd = 256, 8, 100
    drawsType = "mlhs"
    _, drawsMat = makeNormalDraws(nSeq, nDim, drawsType, nInd)