import csv
import numpy as np
import pandas
from scipy import stats
import matplotlib.pyplot as plt

def bootstrap(data, n=5000, func=np.std):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.std(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()

    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],simulations[u_indx])
    return(ci)

v = []
onFirstLine = 1     # used in loop to make sure the column titles are not appended into lists
# read in CSV values into lists
csvfile = open("Oil_Prod.csv")
readCSV = csv.reader(csvfile, delimiter=',')
for row in readCSV:
    if(onFirstLine != 1):
        v.append(int(row[3]))
    onFirstLine = 0

plt.hist(v, bins = 20, range = (0, 10000))
plt.show()

stats.probplot(v,plot=plt)

log_v=np.log(v)
plt.hist(log_v, bins = 20)
plt.show()

stats.probplot(log_v,plot=plt)

boot = bootstrap(v, n=6534)
cintervals = [boot(i) for i in (.90, .95, .99, .995)]

print(cintervals[1])
print(cintervals[2])