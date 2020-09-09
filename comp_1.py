import csv
import numpy as np
import matplotlib.pyplot as plt

def computeFour(list):
    a = np.array(list).astype(np.float)
    print("\t\tMean: " + str(np.mean(a)))
    print("\t\tMedian: " + str(np.median(a)))
    print("\t\tVariance: " + str(np.var(a)))
    print("\t\tStandard Deviation: " + str(np.std(a)))

sepalLengths = []
sepalWidths = []
petalLengths = []
petalWidths = []
varieties = []
onFirstLine = 1     # used in loop to make sure the column titles are not appended into lists

# read in CSV values into lists
csvfile = open("iris.csv")
readCSV = csv.reader(csvfile, delimiter=',')

for row in readCSV:
    if(onFirstLine != 1):
        sepalLengths.append(row[0])
        sepalWidths.append(row[1])
        petalLengths.append(row[2])
        petalWidths.append(row[3])
        varieties.append(row[4])
    onFirstLine = 0

print("All Varieties")
print("\tSepal Length Computations")
computeFour(sepalLengths)
print()

print("\tSepal Width Computations")
computeFour(sepalWidths)
print()

print("\tPetal Length Computations")
computeFour(petalLengths)
print()

print("\tPetal Width Computations")
computeFour(petalWidths)
print()

print()

print("Setosa Variety")
setosaSepalLengths = []
setosaSepalWidths = []
setosaPetalLengths = []
setosaPetalWidths = []
for x in range(50):
    setosaSepalLengths.append(sepalLengths[x])
    setosaSepalWidths.append(sepalWidths[x])
    setosaPetalLengths.append(petalLengths[x])
    setosaPetalWidths.append(petalWidths[x])
print("\tSepal Length Computations")
computeFour(setosaSepalLengths)
print()

print("\tSepal Width Computations")
computeFour(setosaSepalWidths)
print()

print("\tPetal Length Computations")
computeFour(setosaPetalLengths)
print()

print("\tPetal Width Computations")
computeFour(setosaPetalWidths)
print()

print("Versicolor Variety")
versicolorSepalLengths = []
versicolorSepalWidths = []
versicolorPetalLengths = []
versicolorPetalWidths = []
for x in range(50, 100):
    versicolorSepalLengths.append(sepalLengths[x])
    versicolorSepalWidths.append(sepalWidths[x])
    versicolorPetalLengths.append(petalLengths[x])
    versicolorPetalWidths.append(petalWidths[x])
print("\tSepal Length Computations")
computeFour(versicolorSepalLengths)
print()

print("\tSepal Width Computations")
computeFour(versicolorSepalWidths)
print()

print("\tPetal Length Computations")
computeFour(versicolorPetalLengths)
print()

print("\tPetal Width Computations")
computeFour(versicolorPetalWidths)
print()

print("Virginica Variety")
virginicaSepalLengths = []
virginicaSepalWidths = []
virginicaPetalLengths = []
virginicaPetalWidths = []
for x in range(100, 150):
    virginicaSepalLengths.append(sepalLengths[x])
    virginicaSepalWidths.append(sepalWidths[x])
    virginicaPetalLengths.append(petalLengths[x])
    virginicaPetalWidths.append(petalWidths[x])
print("\tSepal Length Computations")
computeFour(virginicaSepalLengths)
print()

print("\tSepal Width Computations")
computeFour(virginicaSepalWidths)
print()

print("\tPetal Length Computations")
computeFour(virginicaPetalLengths)
print()

print("\tPetal Width Computations")
computeFour(virginicaPetalWidths)
print()

a = np.array(petalLengths).astype(np.float)
plt.hist(a, 18)
plt.title("Histogram with All Petal Lengths")
plt.show()

a = np.array(setosaPetalLengths).astype(np.float)
plt.hist(a, 10)
plt.title("Histogram with Setosa Petal Lengths")
plt.show()

a = np.array(versicolorPetalLengths).astype(np.float)
plt.hist(a, 10)
plt.title("Histogram with Versicolor Petal Lengths")
plt.show()

a = np.array(virginicaPetalLengths).astype(np.float)
plt.hist(a, 10)
plt.title("Histogram with Virginica Petal Lengths")
plt.show()

a = np.array(petalLengths).astype(np.float)
plt.boxplot(a)
plt.title("Box and Whisker Plot with All Petal Lengths")
plt.show()

print("Petal Length Quantiles")
print("\t5th Percentile: " + str(np.quantile(a, 0.05)))
print("\t25th Percentile: " + str(np.quantile(a, 0.25)))
print("\t50th Percentile: " + str(np.quantile(a, 0.50)))
print("\t75th Percentile: " + str(np.quantile(a, 0.75)))
print("\t95th Percentile: " + str(np.quantile(a, 0.95)))