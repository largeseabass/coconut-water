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
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance

import time
import warnings
from dbfread import DBF



"""
Some Paths used in testing
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
raster_path_this = path_of_this_folder+'Madagascar_sand.tif'
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
    [csv_path]: string, the path where the csv file containing (all variables + endemicity level) is stored.
    [target_population]: string, in ESPEN database, there are three categories for population.
    Select one category before doing the random forest classification.

    This section needs to be carefully modified when using on database other than ESPEN.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            #warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
            print(1)
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
            clf.fit(X_train_std,y_train)
            y_pred=clf.predict(X_test_std)
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


            return clf, sc, X_test_std,y_pred,x_labels[1:,],np.array(y_test)

    except:
        pass
        print("Random Forest Classifier: Fail")
        print("Please Check the input file format")


def create_file(endemicity_csv_path,saving_csv_path):
    """
    [endemicity_csv_path]: string, the path of the endemicity data downloaded.
    [saving_csv_path]: string, the path to save the master endemicity csv file.

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
    [saving_csv_path]: string, the path for endemicity csv file, which act as the master csv data saving file.
    [new_csv_path]: string, the path for the csv file to be added to the endemicity file.
    [column_taken]: string, the name of the column of the new csv file to be added to the endemicity csv file.
    [key_column]: string, the name of the column(indicating the district) we use to compare with 'name_match' column
    in endemicity csv file to perform dataframe merges.
    [new_column_name]: string, the name of the new column added to the endemicity csv file.

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

def add_raster(raster_list,vector_path,name_list,saving_csv_path,key_column,temporary_path):
    """
    [raster_list]: np.array/list, a list of files to be processed in QGIS zonal statistics.
    [vector_path]: string, the path of the shape file used as mask for zonal STATISTICS.
    [name_list]: np.array/list, a list of column/key names for store the zonal statistics results in the endemicity csv file.
    One name for each raster file in raster_list.
    [saving_csv_path]: string, the path for endemicity csv file, which act as the master csv data saving file.
    [key_column]: string, the name of the column(indicating the district) we use to compare with 'name_match' column
    in endemicity csv file to perform dataframe merges.
    [temporary_path]: string, the path with the name of .csv file to store temporary csv file. The created file will be removed at the end of the process.


    temporary_path is where to store the temporary file. This file will be delete at the end of each iteration.

    For each file in the raster list:
    perform zonal statistics(mean) and then the same operation as in the add_csv.
    """
    try:



        for i in range(len(raster_list)):
            # Run zonal statistics and store the output csv file in the temporary path
            file_output = temporary_path
            file_input = raster_list[i]
            feedback = QgsProcessingFeedback()

            #(A)-start
            processing.run("native:zonalstatisticsfb", {'INPUT':vector_path,'INPUT_RASTER':file_input,'RASTER_BAND':1,'COLUMN_PREFIX':'_','STATISTICS':[2],'OUTPUT':file_output})
            #(A)-end
            
            """
            The code below was added by Michael on 3/30/2022, 
            this requires that the following import statement be added at the start of the script: from dbfread import DBF
            If you have problem running the processing.run(), try comment out (A) and use the codes (B) below.
            """
            #(B)-start
            # processing.run("native:zonalstatistics", {'INPUT_VECTOR':vector_path,'INPUT_RASTER':file_input,'RASTER_BAND':1,'COLUMN_PREFIX':'_','STATISTICS':[2]})
            # csv_fn = temporary_path
            # table = DBF(vector_path[:-4] + ".dbf")# table variable is a DBF object
            # with open(csv_fn, 'w', newline = '') as f:# create a csv file, fill it with dbf content
            #     writer = csv.writer(f)
            #     writer.writerow(table.field_names)# write the column name
            #     for record in table:# write the rows
            #         writer.writerow(list(record.values()))
            #(B)-end
            
            #Combine files - start
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
        print("Please check the saving_csv_path and file format")
        return False


def partial_dependence_plot(X_test, this_clf, n_number,store_graph_path,x_labels):
    """
    [X_test] np.array, the np.array of variables (need to make sure none of them are in the training set)
             ! pre-processed with scalar [see [sc] in def random_forest_modeling]
    [this_clf] machine learning model, the random forest model after training
    [n_number] int, the number of points to take to create partial dependence plot
    [store_graph_path] string, the directory to store the partial dependence plot
    [x_labels] np.array, the label of variables, in the order of X_test.

    for each variable:
        (1) create a copy of [X_test], create empty lists for y_pred, variable_value.
        (2) find the variable's range
        (3) interpolate ([n_number]+2) intervals, create a list of variable values without the two end values.(n_number in total)
        (4) for each variable value:
        (4.1)replace the whole column with that particular value
        (4.2) random forest classification with [this_clf], and store average value of this_y_pred in y_pred
        (5) plot graph for y_pred against variable_value
        (6) store the graph in the directory [store_graph_path] with name given by [x_labels]

    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            xarray = []
            yarray = []
            for i in range(len(x_labels)):
                y_pred = []
                print("PDP:"+x_labels[i])

                #create a copy of X_test and change the value of the list
                this_X = X_test.copy()
                this_key = x_labels[i]
                this_column = this_X[:,i]
                print(this_column)
                variable_value0 = np.sort(this_column)
                print(variable_value0)
                variable_value = []
                #to eliminate extreme values
                index_value = np.floor(len(variable_value0)/(n_number+2)).astype("int")
                m = index_value
                while m <= (len(variable_value0)-1):
                    variable_value.append(variable_value0[m])
                    m = m+index_value

                for this_value in variable_value:
                    this_X[:,i] = this_value
                    #random Forest
                    this_y_pred=this_clf.predict(this_X)
                    y_pred.append(np.average(this_y_pred))

                xarray.append(variable_value)
                yarray.append(y_pred)

                store_graph_name = store_graph_path + x_labels[i]+".png"

                fig, ax = plt.subplots()
                ax.plot(variable_value, y_pred)

                ax.set(xlabel=x_labels[i], ylabel='endemicity')
                ax.set_title('Partial Dependence Plot: Endemicity against '+x_labels[i]+'(after scaling)')
                ax.grid(True)
                plt.savefig(store_graph_name)


            return xarray,yarray

    except:
        print("Partial Dependence Plot: fail")
        print("Please check the store_graph_path")


def plot_test_sample(store_graph_path, x_labels,y_pred,X_test,y_test):
    """
    [store_graph_path] string, the path of the directory to store the graphs.
    [x_labels] np.array, the array of variable labels
    [y_pred] np.array, the array of predicted y value
    [X_test] np.array, the array of variables (Need to make sure none of them are in the training sample)

    Generate plot of predicted endemicity value against
    """
    try:
        for i in range(len(x_labels)):
            store_graph_name = store_graph_path + x_labels[i]+".png"
            print(store_graph_name)
            x1, y1, x2, y2, x3, y3 = [],[],[],[],[],[]
            y11, y22, y33 = [],[],[]
            for j in range(len(y_pred)):
                if y_pred[j] == 1:
                    x1.append(X_test[j][i])
                    y1.append(1)
                    y11.append(y_test[j])
                else:
                    if y_pred[j] == 2:
                        x2.append(X_test[j][i])
                        y2.append(2)
                        y22.append(y_test[j])
                    else:
                        x3.append(X_test[j][i])
                        y3.append(3)
                        y33.append(y_test[j])

            fig, ax = plt.subplots()
            x01 = np.concatenate([x1,x2,x3])
            y01 = np.concatenate([y1,y2,y3])
            y02 = np.concatenate([y11,y22,y33])
            ax.scatter(x01, y01, c='tab:blue', label='test value',alpha=0.5, edgecolors='none')
            ax.scatter(x01, y02, c='tab:orange', label='real value',alpha=0.5, edgecolors='none')
            ax.set(xlabel=x_labels[i], ylabel='endemicity')
            ax.set_title('Predict Endemicity against '+x_labels[i]+'(after scaling)')
            ax.legend()
            ax.grid(True)
            plt.savefig(store_graph_name)
            #
    except:
        print("Plot Test Sample: fail")
        print("Please check store_graph_path, x_label, y_pred, X_test")



def importance_plot(clf, store_graph_path,x_labels,X_test, y_test, n_repeats=10, Use_Permutation=True):
    """
    [clf] the random forest model. Input values need to be scaled.
    [store_graph_path] string, the directory to store the graph.
    [x_labels] np.array, the names of variables.
    [X_test] np.array, the variables after scaling and in the order of x_labels.
    [y_test] np.array, the actual value of endemicity.
    [n_repeats] int, the number of times to sample the variables for permutation importance plot.
    [Use_Permutation] bool, True for performing permutation importance plot, False for performing impurity importance plot.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if Use_Permutation:
            #permutation importance
            print("Permutation Importance Calculation")
            result = permutation_importance(clf, X_test, y_test, n_repeats=n_repeats, random_state=42)
            forest_importances = pd.Series(result.importances_mean, index=x_labels)

            fig, ax = plt.subplots(figsize=(60,20))

            forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
            ax.set_title("Feature importances using permutation on full model")
            ax.set_ylabel("Mean accuracy decrease")
            fig.tight_layout()
            store_graph_name = store_graph_path + "permutation_importance.png"
            plt.savefig(store_graph_name)


        else:
            #impurity importance
            print("Impurity Importance Calculation")
            importances = clf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
            forest_importances = pd.Series(importances, index=x_labels)

            fig, ax = plt.subplots(figsize=(60,20))
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")

            fig.tight_layout()
            store_graph_name = store_graph_path + "impurity_importance.png"
            plt.savefig(store_graph_name)



"""
Some commands use to test
"""
"""
(1)
Test random forest modeling
2014all.csv is the csv file contains all the variable data I've processed.
[this_clf] the random forest model
[this_scalar] the scaling methods
[X_test] the 2d matrix(number of samples * number of variables) used to store the ctx_variables
[x_labels] the array of variable names

"""
# this_clf, this_scalar,X_test,y_pred,x_labels,y_test = random_forest_modeling(csv_path, "Total")


"""
Importance Plot
"""
# store_graph_path = "/Users/huangliting/Desktop/geohelminth_test/mean_decrease_impurity/"
# importance_plot(this_clf, store_graph_path,x_labels,X_test, y_test, n_repeats=10, Use_Permutation=True)

"""
Plot Partial Dependence Plots
n_number is the number of data points in one plot.
"""
# store_graph_path = "/Users/huangliting/Desktop/geohelminth_test/mean_decrease_impurity/"
# n_number = 10
# xarray, yarray = partial_dependence_plot(X_test, this_clf, n_number,store_graph_path,x_labels)


"""
plot a graph of: for each variable, plot a scatter graph for the three level of endemicity
real and predicted endemicity
"""
# store_graph_path = "/Users/huangliting/Desktop/geohelminth_test/plot/"
# plot_test_sample(store_graph_path, x_labels,y_pred,X_test,y_test)



"""
(2)
Test Create csv
create a new endemicity csv file.
"""
# if create_file(endemicity_csv_path,saving_csv_path):
#     print("csv created")


"""
(3)
Test add csv to the endemicity csv
"""
# column_name = 'p2014est'
# key_this_csv = 'ADM2_EN'
# new_column_name = '2014Population'

# if add_csv(saving_csv_path,add_new_csv_path,column_name,key_this_csv,new_column_name):
#     print("Success")


"""
(4)
Test add raster (use zonal statistics to change into csv) to endemicity csv
"""
#
# namelist = ['sand']
# key_this_csv = 'ADM2_EN'
# temporary_path = '/Users/huangliting/Desktop/Zonal_statistics_results/trythis.csv'
# if add_raster(raster_list,vector_path,namelist,saving_csv_path,key_this_csv, temporary_path):
#     print("Success")



"""
Exit QGIS at the end.
"""

qgs.exitQgis()
