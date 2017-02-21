"""
Single-purpose script for plotting the accuracies given by the 
"""

from __future__ import print_function
from matplotlib import pyplot as plt

#took these directly from the R-variant output text files for R=1,10, 25, etc
#accuracies = [67.4,  68.0, 70.0, 71.8, 72.1, 72.1]
accuracies = [68.6,72.1,74.5,76.5,75.0]
plt.title("Test Accuracy")
xs = [i for i in range(0,len(accuracies))]
xlabels = ["1","10","25","50","100"]
plt.xticks(xs, xlabels)
plt.xlabel("Max-iterations")
plt.ylabel("Test-Set Hamming Accuracy %")
plt.plot(xs,accuracies)
plt.savefig("phi3NetAccuracy.png")
plt.show()



