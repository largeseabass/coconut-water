# coconut-water
As clear as coconut-water, some simple codes to make geohelminth infection prediction based on random forest classifier using weather and socioeconomic data.


## Installation Requirement

(1) Need to Download the latest Version of QGIS.


(2) This script may only work on mac.


(3) Please download all the files then read all the lines in the geohelminth_test.py


This guides you through how to set up and use the functions. Tutorials are also included in geohelminth_test.py. Functions are very easy to use.



Make sure you run the Python script using the python in QGIS. (you could set the path in your compiler)


## Data

Endemicity Data downloaded from the ESPEN database.


I'll share a list of all data sources later. They can all be found online. 



## Output

Most Outputs are .csv files.

You can get Partial Dependence Plots using def partial_dependence_plot.

![alt text](https://github.com/largeseabass/coconut-water/blob/main/clay__mean.png)


For Importance Plots generating using def importance_plot, here are two examples:

![alt text](https://github.com/largeseabass/coconut-water/blob/main/impurity_importance.png)

![alt text](https://github.com/largeseabass/coconut-water/blob/main/permutation_importance.png)



To get an overview of how real and predicted endemicity distributed in test data, you can use plot_test_sample.

![alt text](https://github.com/largeseabass/coconut-water/blob/main/clay__mean_test.png)
