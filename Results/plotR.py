"""
Single-purpose script for plotting the accuracies given by the 
"""

from __future__ import print_function
from matplotlib import pyplot as plt

#took these directly from the R-variant output text files for R=1,10, 25, etc
accuracies = [67.4,  68.0, 70.0, 71.8, 72.1, 72.1]
plt.title("Accuracy Per R-Value, Random Uniform Sequences")
xs = [i for i in range(0,len(accuracies))]
xlabels = ["1","10","25","50","100","200"]
plt.xticks(xs, xlabels)
plt.xlabel("R")
plt.ylabel("Hamming Accuracy %")
plt.plot(xs,accuracies)
plt.savefig("r_accuracy.png")
plt.show()



