import numpy as np
import matplotlib.pyplot as plt
kmax = 53
Kmax = 23
beta = 0.01
gamma = 0.8
E = 100736
V = 7006
directoryA = "../singlespreaders/sir/scenario1/groupA"
directoryB = "../singlespreaders/sir/scenario1/groupB"
directoryC = "../singlespreaders/sir/scenario1/groupC"

table =[]


matrice = np.loadtxt('av_sir_results_node2.txt', skiprows=1)

column2 = matrice[:,1][:10] #10
column3 = sum(matrice[:,2])
#print(matrice)
print column2
print column3



