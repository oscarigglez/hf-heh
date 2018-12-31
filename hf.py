#!./bin/python3

import numpy as np

def trimul(A,B,C):
    return np.matmul(np.matmul(A,B),C)

def eint(a,b,c,d):
    # This function condenses all 4 indexes of a two electron integral into one unique value to solve the problem of the multiple equivalences. This clever piece of code was taken directly from https://joshuagoings.com/2013/04/24/hartree-fock-self-consistent-field-procedure/.
    if a > b: ab = a*(a+1)/2 + b
    else: ab = b*(b+1)/2 + a
    if c > d: cd = c*(c+1)/2 + d
    else: cd = d*(d+1)/2 + c
    if ab > cd: abcd = ab*(ab+1)/2 + cd
    else: abcd = cd*(cd+1)/2 + ab
    return abcd

def tei(a,b,c,d):
    return twoe.get(eint(a,b,c,d), 0.0)

def P_make(F):
    # Generate a new density matrix from a Fock matrix
    F_p = trimul(X,F,X)
    C = np.matmul(X,np.linalg.eigh(F_p)[1])
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            tp = 0
            for a in range(N//2):
                tp += C[i,a]*C[j,a]
            P[i,j] = tp
    return P

def F_make(P):
    # Generate a new Fock matrix from a density matrix
    F = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            tp = 0
            for k in range(N):
                for l in range(N):
                    tp += P[k,l]*(2*tei(i+1,j+1,l+1,k+1) - tei(i+1,k+1,l+1,j+1))
            F[i,j] = Hcore[i,j] + tp
    return F

def mo_fy(M):
    moM = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            tp = 0
            for k in range(N):
                for l in range(N):
                    tp += C[k,i]*M[k,l]*C[l,j]
            moM[i,j] = tp
    return moM

def energy(P,F):
    # Calculate the energy from a pair of density and Fock matrices as described in the exercise
    E = 0
    for i in range(N):
        for j in range(N):
            E += P[i,j]*(Hcore[i,j]+F[i,j])
    return E

twoe = {eint(1,1,1,1) : 1.056248,
        eint(2,1,1,1) : 0.467681,
        eint(2,1,2,1) : 0.246491,
        eint(2,2,1,1) : 0.606397,
        eint(2,2,2,1) : 0.388706,
        eint(2,2,2,2) : 0.774998}

S = np.array([[1.0, 0.560586], [0.560586, 1.0]])
U = np.array([[1, -1], [1, 1]])
s = np.diag(np.linalg.eigh(S)[0])
s05 = np.diag(np.linalg.eigh(S)[0]**(-0.5))
U = np.linalg.eigh(S)[1]
Uinv = np.linalg.inv(U)
X = trimul(U,s05,Uinv)

N=2
its = 10

P_0 = np.array([[1, 0],[0, 0]])
Hcore = np.array([[-2.644382, -1.511774],[-1.511774, -1.778216]])

F = F_make(P_0)
E = energy(P_0,F)
print("Iteration 0: {0:12.10f}".format(E+1.428571))
i = 0
deltaE = 1
while deltaE > 0.000001:
    i += 1
    lastE = E
    P = P_make(F)
    F = F_make(P)
    E = energy(P,F)
    deltaE = abs(E-lastE)
    print("Iteration {0:d}: {1:12.10f}    deltaE: {2:12.10f}".format(i, E+1.428571, deltaE))

F_p = trimul(X,F,X)
C = np.matmul(X,np.linalg.eigh(F_p)[1])
moF = mo_fy(F)
print(moF)
moP = mo_fy(P)
print(moP)
