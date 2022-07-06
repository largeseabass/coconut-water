# coconut-water
As clear as coconut-water, some simple codes to make geohelminth infection prediction based on random forest classifier using weather and socioeconomic data.


## Installation Requirement

This script may only work on mac.

(1) Need to Download the latest Version of QGIS. (I'm using QGIS 3.24.0-Tisler)

(2) Clone this responsitory (can ignore the /results file)


Reading all the lines in the geohelminth_test.py guides you through how to set up and use the functions. Tutorials are also included in geohelminth_test.py. Functions are very easy to use.



Make sure you run the Python script using the python in QGIS. (you could set the path in your compiler)


## Install GDAL


<Mac> via conda in terminal:


(1) Create a new conda enviornment


(2) Conda-forge GDAL


(3) Change Path


(4) Check GDAL version



'''
conda create --name  gis-gdal
conda activate gis-gdal   
conda install -n gis-gdal -c conda-forge gdal=3.5.1
echo 'export PATH=/Library/Frameworks/GDAL.framework/Programs:$PATH' >> ~/.bash_profile
source ~/.bash_profile
gdalinfo --version
'''



## Method

This simulation uses Random Forest Classifier(RFC) as the core modeling algorithm. Other functions in this script are used for converting raw data(csv or raster) into the format(a csv file containing all the input variables + endemicity) which could be processed by RFC.


### Convert Raster into csv file

Use QGIS zonal_statistics function to calculate the mean value of raster data in boundaries defined by vector data. 


### Pre-process training data in RFC

(1) Splitting data into training and testing sets.


(2) Here, our geohelminth infection data for year 2014 is unbalanced. We have 3 classes of infection data: low, medium and high endemicity. 


For the training set: We count which class has the highest number, then randomly duplicate members of the other two groups to make their number equals to the highest number.


Testing set remains unchanged.


(3) For the variables: we use a scalar to get all the variables in training set into the same range. Then this scalar is applied to the testing set.

(4) Train the Model using training set. 

(5) Test the Model using testing set. 


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



## Problem & Improvement


(1) Could use GDAL to replace QGIS.

(2) Could use new methods to split training and testing sets. Here it is randomly splitted.

(3) New methods to treat unbalanced data?

(4) New variables?





