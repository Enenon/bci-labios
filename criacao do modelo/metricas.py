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



        



