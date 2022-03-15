"""
This is a master file for all the functions we need.

The Design Idea is:
(1) Use Endemicity data to create a data-storage csv file.
def create_file(string endemicity-csv-path, string saving-csv-path); return a boolean indicating success/fail
(2) Add either csv file or raster file to the data-storage csv file each time a new variable is considered.
def add_csv(string saving-csv-path, string this_csv_path); return a boolean indicating success/fail
def add_raster(string saving-csv-path, string this_raster_path, string this_vector_path); return a boolean indicating success/fail
(3) When all the variables are added, perform Random Forest Analysis.
def random_forest_modeling(string); print the accraucy and truth table, return the random_forest model and standard scalar


Notes:
(1) For geohelminths in Madagascar, the lower two endemicity classes are mixed.
(2) If using database other than ESPEN, need to change target-population session in random_forest_modeling


"""

import pandas as pd
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample, shuffle
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import os

import time

"""
Some Paths used in testing

Change path_of_this_folder to the one stored the downloaded folder.
"""
path_of_this_folder = '/Users/huangliting/Desktop/geohelminth_test/'


#Use this csv to test random forest model
csv_path = path_of_this_folder+'2014all.csv'
#Use this to test create csv file for endemicity data each year
saving_csv_path = path_of_this_folder+'test.csv'
endemicity_csv_path = path_of_this_folder+'geohelminth_infection_2014.csv'

#Use this to test add new column from csv file to storage csv file
add_new_csv_path = path_of_this_folder+'mdg_pop_adm2.csv'

#Test Zonal STATISTICS
raster_path_this = path_of_this_folder+'VNL_v2_npp_2014_global_vcmslcfg_c202101211500.average.tif'
raster_list = [raster_path_this]
vector_path = path_of_this_folder+'mdg_admbnda_adm2_BNGRC_OCHA_20181031.shp'




"""
This Section is for setting up QGIS python console.
This shall be removed after changing QGIS to GDAL.

[BEFORE RUNNING THE SCRIPT]

(1)Go to QGIS Plugins-PythonConsole. In the command line, type:
    import os
    import sys
    print(os.environ)
    print(sys.path)
Using the results to fill-in 'env' and 'paths'

(2) Go to folder to check if the paths for the following os.environ settings are correct.
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
os.environ['DYLD_INSERT_LIBRARIES']
os.environ['PYTHONPATH']

Probelms with QGIS can often be solved by re-installing the newest version of QGIS.

(3) Set QgsApplication.setPrefixPath

(4) Make sure this is excecuting using the python3.9 in the QGIS console.
(e.g. /Applications/QGIS.app/Contents/MacOS/bin/python3.9 {file})
you may want to copy the random_forest session to a separate python file and run it using your default python.
Random Forest can still work using QGIS python, but the warnings are annoying.
"""

import os
import sys
import subprocess


env = {'USER': 'huangliting', 'MallocNanoZone': '0', '__CFBundleIdentifier': 'org.qgis.qgis3', 'COMMAND_MODE': 'unix2003', 'LOGNAME': 'huangliting', 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'PYQGIS_STARTUP': 'pyqgis-startup.py', 'SSH_AUTH_SOCK': '/private/tmp/com.apple.launchd.eUuD4OkY4J/Listeners', 'SHELL': '/bin/zsh', 'MallocSpaceEfficient': '0', 'HOME': '/Users/huangliting', 'QT_AUTO_SCREEN_SCALE_FACTOR': '1', '__CF_USER_TEXT_ENCODING': '0x1F5:0x0:0x2', 'TMPDIR': '/var/folders/lh/t7fjqv3j791d2wjbckpwg5480000gn/T/', 'XPC_SERVICE_NAME': 'application.org.qgis.qgis3.38543067.38544250', 'XPC_FLAGS': '0x0', 'GDAL_DRIVER_PATH': '/Applications/QGIS.app/Contents/MacOS/lib/gdalplugins', 'GDAL_DATA': '/Applications/QGIS.app/Contents/Resources/gdal', 'PYTHONHOME': '/Applications/QGIS.app/Contents/MacOS', 'GDAL_PAM_PROXY_DIR': '/Users/huangliting/Library/Application Support/QGIS/QGIS3/profiles/default/gdal_pam/', 'GISBASE': '/Applications/QGIS.app/Contents/MacOS/grass', 'GRASS_PAGER': 'cat', 'LC_CTYPE': 'UTF-8', 'SSL_CERT_DIR': '/Applications/QGIS.app/Contents/Resources/certs', 'SSL_CERT_FILE': '/Applications/QGIS.app/Contents/Resources/certs/certs.pem'}

paths = ['/Applications/QGIS.app/Contents/MacOS/../Resources/python', '/Users/huangliting/Library/Application Support/QGIS/QGIS3/profiles/default/python', '/Users/huangliting/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins', '/Applications/QGIS.app/Contents/MacOS/../Resources/python/plugins', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/opencv_contrib_python-4.3.0.36-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/patsy-0.5.1-py3.9.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/statsmodels-0.11.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/matplotlib-3.3.0-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/numba-0.50.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/rasterio-1.1.5-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/Pillow-7.2.0-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/pandas-1.3.3-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/geopandas-0.8.1-py3.9.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python39.zip', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/cftime-1.2.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/GDAL-3.3.2-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/netCDF4-1.5.4-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/numpy-1.20.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/Fiona-1.8.13.post1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/Rtree-0.9.7-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/pyproj-3.2.0-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/site-packages/scipy-1.5.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS.app/Contents/MacOS/lib/python3.9/lib-dynload', '/Users/huangliting/Library/Application Support/QGIS/QGIS3/profiles/default/python']

for k,v in env.items():
    os.environ[k] = v

for p in paths:
    sys.path.insert(0,p) #insert the p at the front of list of the path


os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/Applications/QGIS.app/Contents/PlugIns'

os.environ['DYLD_INSERT_LIBRARIES'] = '/Applications/QGIS.app/Contents/MacOS/lib/libsqlite3.dylib'

os.environ['PYTHONPATH'] = '/Applications/QGIS.app/Contents/MacOS/bin/python3.9'


from qgis.core import *
from qgis.utils import *
from qgis.gui import *
from qgis.PyQt import QtGui

qgs = QgsApplication([], False)
QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS",True)
print("Ready")
qgs.initQgis()
import processing

from processing.core.Processing import Processing

from qgis.analysis import QgsNativeAlgorithms


QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
print("Processing")


"""
[Functions]
"""


def random_forest_modeling(csv_path, target_population):
    """
    <1> Read Files
    """

    #Population selection: PreSAC, SAC, Total
    pop_items_list = ['PreSAC', 'SAC', 'Total']
    if target_population in pop_items_list:
        target_pop = target_population
    else:
        target_pop = 'Total'
        print("Target Population tag not applicable, use total population as defualt")


    df = pd.read_csv(csv_path)

    df.loc[df['TargetPop'] == target_pop]

    """
    <2> Relabel the Endemicity

    #List the names of keys here
    #Class 1: EVPSOIL, EVPTRNS, GRN,  GWETPROF, GWETROOT, GWETTOP, PRECTOTLAND,QINFIL
    #Class 2: TSOIL1, TSOIL2, TSOIL3, TSOIL4, TSOIL5, TSOIL6
    #Class 3: 'waterPH_mean', 'bulkDensity__mean', 'clay__mean', 'coarsefragment__mean', 'sand__mean', 'silt__mean', 'Population','night'

    # label_class_1 = ['EVPSOIL', 'EVPTRNS', 'GRN',  'GWETPROF', 'GWETROOT', 'GWETTOP', 'PRECTOTLAND','QINFIL','EVP','GWE','TSOIL']
    # label_class_3 = ['waterPH_mean', 'bulkDensity__mean', 'clay__mean', 'coarsefragment__mean', 'sand__mean', 'silt__mean', 'Population','night_mean']
    # label_class_soil = ['waterPH_mean', 'bulkDensity__mean', 'clay__mean', 'coarsefragment__mean', 'sand__mean', 'silt__mean']
    """

    # Get indexes where name column has value john
    indexNames = df[df['Endemicity'] == 'Not reported'].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)


    #Combine the "Non-endemic" and "Low Prevalence" classes, since the "Non-endemic" group contains way fewer cases than the other ones.
    df.loc[df['Endemicity'] == 'Non-endemic', 'Endemicity'] = 1
    df.loc[df['Endemicity'] == 'Low prevalence (less than 20%)', 'Endemicity'] = 1
    df.loc[df['Endemicity'] == 'Moderate prevalence (20%-49%)', 'Endemicity'] = 2
    df.loc[df['Endemicity'] == 'High prevalence (50% and above)', 'Endemicity'] = 3

    print("[Endemicity Counts]")
    print(df['Endemicity'].value_counts())



    """
    <3> RF_calculation functions
    Use RF_calculation

    """
    y=df['Endemicity']  # Get the y values
    y=y.astype('int')

    #useless labels now: ['Unnamed: 0','name_match','TargetPop','Cov'])
    df.drop(['Unnamed: 0','name_match','TargetPop','Cov'], axis=1,inplace=True)
    x_labels = np.array(df.keys())

    X = df[df.keys()]
    X.astype('float')

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    # do something with unbalanced sample based on the value of'Endemicity'
    X_test.drop(['Endemicity'],axis=1,inplace=True)

    # set the minority class to a separate dataframe
    df_1 = X_train[X_train['Endemicity'] == 1]
    df_2 = X_train[X_train['Endemicity'] == 2]
    df_3 = X_train[X_train['Endemicity'] == 3]

    print(X_train['Endemicity'].value_counts())

    find_max_value = max(np.array(X_train['Endemicity'].value_counts()))

    #upsample the minority classes.
    #If one class is the majority class, it skips this process.
    if (df_1.shape[0] == find_max_value):
        df_1_upsampled = df_1
        print("Low Endemicity has the most samples here")
    else:
        df_1_upsampled = resample(df_1, random_state=42, n_samples=find_max_value, replace = True)

    if (df_2.shape[0] == find_max_value):
        df_2_upsampled = df_2
        print("Moderate Endemicity has the most samples here")
    else:
        df_2_upsampled = resample(df_2, random_state=42, n_samples=find_max_value, replace = True)

    if (df_3.shape[0] == find_max_value):
        df_3_upsampled = df_3
        print("High Endemicity has the most samples here")
    else:
        df_3_upsampled = resample(df_3, random_state=42, n_samples=find_max_value, replace = True)

    #concatenate the upsampled dataFrame
    X_train = pd.concat([df_2_upsampled, df_1_upsampled,df_3_upsampled])

    print("[Check the values after up-sample]")
    print(X_train['Endemicity'].value_counts())

    #Clean up the datasets after upsampling + double-check their types

    X_train = X_train.astype('float')
    X_test = X_test.astype('float')

    y_train = X_train['Endemicity'].copy()
    y_train = y_train.astype('int')

    X_train.drop(['Endemicity'],axis=1,inplace=True)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)


    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=1000)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #y_score = clf.predict_proba(X_test)
    #print("Random Forest Score")
    #print(y_score)

    print("[Random Forest Results]")
    score_this = accuracy_score(y_test, y_pred)
    print("Accuracy"+str(score_this))
    #print(metrics.confusion_matrix(y_test, y_pred))
    confusion_matrix_this = metrics.confusion_matrix(y_test, y_pred)

    for i in range(3):
        print("[Actual Endemicity Level %i ] [Predict 1: %i] [Predict 2: %i] [Predict 3: %i]"%((i+1),confusion_matrix_this[i,0],confusion_matrix_this[i,1],confusion_matrix_this[i,2]))



    return clf, sc


def create_file(endemicity_csv_path,saving_csv_path):
    """
    [endemicity_csv_path]: the path of the endemicity data downloaded.
    [saving_csv_path]: the path to save the master endemicity csv file.

    Process the endemicity data file downloaded. Take only the columns we are interested.
    district name + endemicity + cov + target population
    Save the results in csv format, and this endemicity csv file acts as the master data storage csv file.
    We can later add the environment/socioeconomic variable for each district to this master endemicity csv file.

    For convenience, change the key/column name for district name to 'name_match'.
    """
    try:
        #Read infection data(directly downloaded from ESPEN database)
        data_inf = pd.read_csv(endemicity_csv_path)
        #delete columns we don't want
        data_inf.drop(['CONTINENT', 'REGION', 'WHO_REGION', 'ADMIN0', 'ADMIN0ID', 'ADMIN0_FIP',
               'ADMIN0ISO2', 'ADMIN0ISO3', 'ADMIN1', 'ADMIN1ID', 'ADMIN2', 'ADMIN2ID',
               'Alt_ADMIN2', 'ADMIN3ID', 'IUs_ADM', 'IUs_NAME', 'IU_ID',
               'IU_CODE', 'Year', 'PopReq', 'PopTrg',
               'PopTreat', 'PC', 'EffPC', 'PC_n', 'EffPC_n', 'EpiCov', 'EpiPC',
               'EpiEffPC', 'EpiPC_n', 'EpiEffPC_n', 'MDA_scheme', 'Endemicity_ID'], axis=1,inplace=True)
        #rename the matching column to name_match
        data_inf.rename(columns={'ADMIN3':'name_match'}, inplace=True)
        #print(data_inf.keys())
        data_inf.to_csv(saving_csv_path)
        print("Create csv datafile: Success")
        return True
    except:
        print("Create csv datafile: Fail")
        print("Please Check the paths and file format")
        return False


def add_csv(saving_csv_path,new_csv_path,column_taken,key_column,new_column_name):
    """
    [saving_csv_path]: the path for endemicity csv file, which act as the master csv data saving file.
    [new_csv_path]: the path for the csv file to be added to the endemicity file.
    [column_taken]: the name of the column of the new csv file to be added to the endemicity csv file.
    [key_column]: the name of the column(indicating the district) we use to compare with 'name_match' column
    in endemicity csv file to perform dataframe merges.
    [new_column_name]: the name of the new column added to the endemicity csv file.

    Add the column chosen from new csv file(stored at new_csv_path) to the old csv file(stored at saving_csv_path),
    with the key name of matching column specify as key_column,
    and the column added has title(key) column_taken.
    We change the old column name 'column_taken' to the new name 'new_column_name'.
    Note: the key in the saving_csv is set to be 'name_match' when creating that saving csv file.
    """
    try:
        #Read Files
        data_add = pd.read_csv(new_csv_path)
        data_saving = pd.read_csv(saving_csv_path)

        #Get the set of unwanted keys
        keyset_add = data_add.keys().drop([key_column,column_taken])
        #Drop these unwanted columns
        data_add.drop(columns=keyset_add, inplace=True)
        #Rename the columns so we can perform merging later
        data_add.rename(columns={key_column:'name_match'},inplace=True)
        #Rename the column we are interested in
        data_add.rename(columns={column_taken:new_column_name},inplace=True)
        #Merge two datasets
        df = pd.merge(data_add, data_saving, on='name_match')
        #save
        df.drop(['Unnamed: 0'], axis=1,inplace=True)
        df.to_csv(saving_csv_path)
        print("Add new csv file: Success")
        return True
    except:
        print("Add new csv file: Fail")
        print("Please Check the new csv file format")
        return False

def add_raster(raster_list,vector_path,name_list,saving_csv_path,key_column):
    """
    [raster_list]: a list of files to be processed in QGIS zonal statistics.
    [vector_path]: the path of the shape file used as mask for zonal STATISTICS.
    [name_list]: a list of column/key names for store the zonal statistics results in the endemicity csv file.
    One name for each raster file in raster_list.
    [saving_csv_path]: the path for endemicity csv file, which act as the master csv data saving file.
    [key_column]: the name of the column(indicating the district) we use to compare with 'name_match' column
    in endemicity csv file to perform dataframe merges.


    temporary_path is where to store the temporary file. This file will be delete at the end of each iteration.

    For each file in the raster list:
    perform zonal statistics and then the same operation as in the add_csv.
    """
    try:

        temporary_path = '/Users/huangliting/Desktop/Zonal_statistics_results/trythis.csv'

        for i in range(len(raster_list)):
            # Run zonal statistics and store the output csv file in the temporary path
            file_output = temporary_path
            file_input = raster_list[i]
            feedback = QgsProcessingFeedback()
            processing.run("native:zonalstatisticsfb", {'INPUT':vector_path,'INPUT_RASTER':file_input,'RASTER_BAND':1,'COLUMN_PREFIX':'_','STATISTICS':[2],'OUTPUT':file_output})

            #Real start
            data_add = pd.read_csv(file_output)
            data_saving = pd.read_csv(saving_csv_path)

            #Get the set of unwanted keys
            keyset_add = data_add.keys().drop([key_column,'_mean'])
            #Drop these unwanted columns
            data_add.drop(columns=keyset_add, inplace=True)
            #Rename the columns so we can perform merging later
            data_add.rename(columns={key_column:'name_match'},inplace=True)
            #Rename the column we are interested in
            data_add.rename(columns={'_mean':name_list[i]},inplace=True)
            #Merge two datasets
            df = pd.merge(data_add, data_saving, on='name_match')
            #save
            df.drop(['Unnamed: 0'], axis=1,inplace=True)
            df.to_csv(saving_csv_path)
            os.remove(file_output)

        print("Add Raster Layer: Success")
        return True

    except:
        pass
        print("Add Raster Layer: Fail")
        print("Please check the file paths and file format")
        return False



"""
Some commands use to test
"""
"""
(1) Test random forest modeling
2014all.csv is the csv file contains all the variable data I've processed.
"""

this_clf, this_scalar = random_forest_modeling(csv_path, "Total")

"""
(2) Test Create csv
create a new endemicity csv file.
"""
# if create_file(endemicity_csv_path,saving_csv_path):
#     print("csv created")


"""
(3) Test add csv to the endemicity csv
"""
# column_name = 'p2014est'
# key_this_csv = 'ADM2_EN'
# new_column_name = '2014Population'
# if add_csv(saving_csv_path,add_new_csv_path,column_name,key_this_csv,new_column_name):
#     print("Success")


"""
(4) Test add raster (use zonal statistics to change into csv) to endemicity csv
"""

# namelist = ['2014night']
# key_this_csv = 'ADM2_EN'
#
# if add_raster(raster_list,vector_path,namelist,saving_csv_path,key_this_csv):
#     print("Success")



"""
Exit QGIS at the end.
"""

qgs.exitQgis()
