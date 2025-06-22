from numpy import zeros

def S(matriz):
    som = [sum(i) for i in matriz]
    return som

def C(matriz):
    cs = zeros(len(matriz))
    for i in range(len(matriz)):
        x = 0
        y = 0
        z = 0
        for j in range(len(matriz)):
            y = y + matriz[i][j]
            z = z + matriz[i][j]**2
            for k in range(len(matriz)):
                x = x + matriz[i][j]*matriz[j][k]*matriz[k][i]
        cs[i] = x/(y**2 + z)
    return cs

def I(matriz):
    iss = [[1/i if i != 0 else 0 for i in j] for j in matriz]
    ii = [sum(j)/(len(iss)-1) for j in iss]
    return ii

def corrPearson(matriz):
    matriz = np.transpose(matriz)
    m = np.zeros((len(matriz),len(matriz)))
    meanM = [np.mean(matriz[i,:]) for i in range(len(matriz))]
    for x in range(len(m)):
        for y in range(len(m)):
            xm, ym = matriz[x,:], matriz[y,:]
            xx = xm - meanM[x]
            yy = ym - meanM[y]
            m[x,y] = round(sum((xx)*(yy))/(np.sqrt(sum(xx*xx)) * np.sqrt(sum(yy*yy))),3)
    return m

def randomThreshold(corr,matriz):
    smatriz = matriz.copy() # por causa da falta desse .copy() eu tava tendo meio mundo de dor de cabeça
    for i in range(len(smatriz[0])):
        smatriz[:,i] = np.random.permutation(smatriz[:,i])
    return corr(smatriz)

def rede(corr,threshold,matriz):
    rnd = threshold(corr,matriz)
    # tirando os 1 da diagonal
    for i in range(len(rnd)): rnd[i,i] = 0
    maxim = rnd.max()
    rmatriz = np.array([[i if i > maxim else 0 for i in j] for j in corr(matriz)])
    return rmatriz

    

import numpy as np
data_dir = '../'

with np.load(r'C:\Users\Enenon\Documents\GitHub\bci\T1l.npz') as xx:
    for item in xx.files:
        x = xx[item]

dado = x[:,:721].transpose()

print(rede(corrPearson,randomThreshold,dado))


