# A data analysis tool
import pandas as pd

#  A scientific computing package
import numpy as np 

# A graphical plotting module. Can generate plots, histograms, bar charts, scatterplots etc
import matplotlib.pyplot as plt 

# PyLab is a module in Matplotlib. PyLab combines the numerical module numpy with - 
  # the graphical plotting module pyplot. 

from pylab import plot, show, bar

# The Main Method
def main():

    
    # Example 1

    X = [590,540,740,130,810,300,320,230,670,620,770,250] #Avg Monthly Med Expenditure
    Y = [32,36,39,52,82,22,27,45,68,57,48,54] #Persons Age
  

    plt.xlabel('Avg Monthly Expenditure')
    plt.ylabel('Persons Age')
    plt.title('Relationship Between Persons Age and Avg Monthly Expenditure')

  
    plt.scatter(X,Y)
    plt.show()


if __name__ == '__main__':
  main()