import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats  import poisson
from scipy.stats import binom
import sys

sys.stdout = open('hybrid1results.txt', 'w')

poissonSimulations = []
tempSum = 0
averages = []
for s in range(10):
    print("Poisson Simulation " + str(s+1) + ":")
    data_poisson = poisson.rvs(mu = 3, size = 100)
    poissonSimulations.append(data_poisson)
    for x in range(100):
        print(str(data_poisson[x]))
        tempSum += data_poisson[x]
    print("Average of Sim " + str(s+1) + ": " + str(tempSum/100))
    print()
    averages.append(tempSum/100)
    tempSum = 0
    
sumAverages = 0
for i in range(len(averages)):
    sumAverages += averages[i]
print("Average of the averages of each Poisson Simulation: " + str(sumAverages/len(averages)))
a = np.array(averages).astype(np.float)
print("Standard deviation of the averages of each Poisson Simulation: " + str(np.std(a)))
print("Difference between mean value and expected value (3.0): " + str(abs(3-sumAverages/len(averages))))

binomSimulations = []
tempSumBinom = 0
averagesBinom = []
for s in range(10):
    print("Binomial Simulation " + str(s+1) + ":")
    data_binom = []
    for x in range(100):
        data_binom.append(binom.rvs(poissonSimulations[s][x], 0.1))
        print(str(data_binom[x]))
        tempSumBinom += data_binom[x]
    binomSimulations.append(data_binom)
    print("Average of Sim " + str(s+1) + ": " + str(tempSumBinom/100))
    print()
    averagesBinom.append(tempSumBinom/100)
    tempSumBinom = 0

sumAveragesBinom = 0
for i in range(len(averagesBinom)):
    sumAveragesBinom += averagesBinom[i]
print("Average of the averages of each Binomial Simulation: " + str(sumAveragesBinom/len(averagesBinom)))
b = np.array(averagesBinom).astype(np.float)
print("Standard deviation of the averages of each Binomial Simulation: " + str(np.std(b)))
print("Difference between mean value and expected value (0.3): " + str(abs(0.3-sumAveragesBinom/len(averagesBinom))))