# %%--  Import modules
#   Package classes
from LumiNet import *
from LumiNet import SaveObj, LoadObj, print_dic
#   Other packages
import pandas as pd
import numpy as np
from torchvision import transforms, models
import datetime

#   Sklearn models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
# %%-
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Set-up
#///////////////////////////////////////////
# %%--  Loading models
Obj = [ #Select one
    LoadObj("Models\\","LumiNet_VGG_RF_Eff"), # Efficiency predictor
    # LoadObj("Models\\","LumiNet_VGG_RF_Isc"), # Current predictor
    # LoadObj("Models\\","LumiNet_VGG_RF_Voc"), # Voltage predictor
    ][0]
#  <subcell>    Object layout
#print_dic(Obj)
'''
Use print_dic to see the nested structure of the loaded dictionary object which looks like this :
( # comments for explanation)
 CNN
	 results
		 Reference
		 Training ID
		 Total training time (s)
		 Average training time per epoch
		 Final training score
		 Best training score
		 Final testing score
		 Best testing score
		 Weighted Accuracy
		 Weighted F1-score
		 Weighted Precision
		 Weighted Recall
		 Recall - class X # for all classes
		 Precision - class X # for all classes
		 F1-score - class X # for all classes
		 AUC - class X # for all classes
		 Macro AUC
	 vocab # shows all classes
		 X
	 CM # Testing confusion matrix in the form of a pandas dataframe
	 lossPlots # training and testing losses over training epochs
	 model_classifier # Deep learning model (pytorch) with feature extractor and classification layer
	 model_extractor # Deep learning model (pytorch) with feature extractor and WITHOUT classification layer
 ML
	 results
		 Reference
		 Training ID
		 Training time (s)
		 Train set score
		 Test set score
		 Train set RMSE
		 Test set RMSE
	 model # Machine learning model (sklearn)
 TIMESTAMP
 PARAMETERS # Parameters used for training
	 NAME
	 SAVE
	 RANDOM_SEED
	 DATASET_ID
	 TARGET_COL
	 TARGET_MOD
	 TARGET_MOD_VAL
	 TARGET_BIN
	 TARGET_BIN_CUSTOM
	 TARGET_BIN_CUSTOM_CLIP
	 ML_FRAC
	 CALCULATE_STATPARAM
	 CALCULATE_DATASET_STAT
	 SAVEFOLDER
	 CNN
		 IMAGE_SIZE
		 PRETRAIN
		 REQGRAD
		 SUBSET_SIZE
		 BATCH_SIZE
		 SPLIT_FRAC
		 N_EPOCHS
		 CM_FZ
		 MODEL
		 TRANSFORM_AUG
		 TRANSFORM
		 NCLASS
	 ML
		 MODEL
		 SUBSET_SIZE
		 SPLIT_FRAC
		 LINEAR
'''
#  </subcell>
TF_model = Obj['CNN']['model_extractor']
# %%-
# %%--  Parameters
PARAMETERS = Obj['PARAMETERS'] # Use loaded parameters as default. Explore PARAMETERS to edit
PARAMETERS['NAME']="FEATURE EXTRACTION TEST"
PARAMETERS['SAVEFOLDER']="TEST\\"
PARAMETERS['TARGET_COL']="Eff"
LOAD_FILE="test.csv" #Should have a "path" column and a TARGET_COL column
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////
# %%--  Load custom dataset
dataset = Datahandler(PARAMETERS['SAVEFOLDER'])
dataset.matchDf = pd.read_csv(PARAMETERS['SAVEFOLDER']+LOAD_FILE)
#  <subcell>    Extract statistical parameters
if PARAMETERS['CALCULATE_STATPARAM']:
    print("Calculating Statistical Parameters")
    print("\t Processing... ")
    dataset.computeStatParameters(threshold=20)
    print("Done\n")
#  </subcell>
#  <subcell>    Calculate dataset statistical parameters
if PARAMETERS['CALCULATE_DATASET_STAT']:
    print("Calculating Dataset Statistics")
    tab=[]
    columns=["Id","Min","Max","Mean","Median","Std","Mad","Min_norm","Max_norm","Mean_norm","Median_norm","Std_norm","Mad_norm","Min_mmad","Max_mmad","Mean_mmad","Median_mmad","Std_mmad","Mad_mmad"]
    print("\t Processing... ")
    if PARAMETERS['TARGET_MOD_VAL'] is not None:
        line = dataset.computerDatasetStats(PARAMETERS['TARGET_COL'], fMean = PARAMETERS['TARGET_MOD_VAL'][0], fStd=PARAMETERS['TARGET_MOD_VAL'][1])
    else:
        line = dataset.computeDatasetStats(PARAMETERS['TARGET_COL'])
    tab.append(line)
    dataset_stat = pd.DataFrame(tab)
    dataset_stat.columns=columns
    print(dataset_stat.to_string())
    if PARAMETERS['SAVE']: dataset_stat.to_csv(PARAMETERS['SAVEFOLDER']+"datasetStatistic_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_"+PARAMETERS['TARGET_COL']+".csv", index=False)
    print("Done\n")
#  </subcell>
#  <subcell>    Modify Target
if PARAMETERS['TARGET_MOD']=='NORM':
    print("Modifying target column using 'NORM' approach")
    print("\t Processing... ")
    dataset.normCol(PARAMETERS['TARGET_COL'], fMean = PARAMETERS['TARGET_MOD_VAL'][0], fStd=PARAMETERS['TARGET_MOD_VAL'][1])
    targetCol=PARAMETERS['TARGET_COL']+'_std_norm'
    print("Done\n")
elif PARAMETERS['TARGET_MOD']=='MMAD':
    print("Modifying target column using 'MMAD' approach")
    print("\t Processing... ")
    dataset.mmadCol(PARAMETERS['TARGET_COL'], fMean = PARAMETERS['TARGET_MOD_VAL'][0], fStd=PARAMETERS['TARGET_MOD_VAL'][1])
    targetCol=PARAMETERS['TARGET_COL']+'_std_mmad'
    print("Done\n")
else:
    print("Not modifying target column")
    targetCol=PARAMETERS['TARGET_COL']
#  </subcell>
#  <subcell>    Cell Binning and define nClass
print("Creating Bins for "+targetCol)
print("Processing... ","\t Binning "+PARAMETERS['TARGET_BIN'])
df=dataset.matchDf
min = np.floor(df[targetCol].min())
max = np.ceil(df[targetCol].max())
if PARAMETERS['TARGET_BIN'] == "10_QUANTILES" :
    S1,S2,S3,S4,S5,S6,S7,S8,S9 = df[targetCol].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    binTab=[-99,S1,S2,S3,S4,S5,S6,S7,S8,S9,max]
elif PARAMETERS['TARGET_BIN'] == "4_QUANTILES" :
    Q1,med,Q3 = df[targetCol].quantile([0.25,0.5,0.75])
    mad = df[targetCol].mad()
    binTab =[-99,Q1,med,Q3,max]
elif PARAMETERS['TARGET_BIN'] == "CUSTOM" :
    binTab=[-np.infty]+PARAMETERS['TARGET_BIN_CUSTOM']
    if min < np.min(binTab):
        print("\t Warning - dataset minimum lower than custom bin min")
        if not PARAMETERS['TARGET_BIN_CUSTOM_CLIP']: binTab[0]=min
    if max > np.max(binTab):
        print("\t Warning - dataset maximum higher than custom bin max")
        if not PARAMETERS['TARGET_BIN_CUSTOM_CLIP']: binTab[-1]=max
else:
    print("\t Warning - No defined binning strategy")
    binTab = []

nClass = len(binTab)-1
PARAMETERS['CNN']['NCLASS']=nClass
dataset.addLabels(targetCol,binTab)
print("Done\n")
#  </subcell>
#  <subcell>    Plot targetCol distribution and binning
dataset.plotCol(targetCol,save=PARAMETERS['SAVE'])
#  </subcell>
# %%-
# %%--  Feature extraction
Xcols, dataset.matchDf = Ptcompute.extractFeature(TF_model,dataset.matchDf,PARAMETERS['CNN']['TRANSFORM'],batch_size=PARAMETERS['CNN']['BATCH_SIZE'])
# %%-
# %%--  Machine learning regression
Skmodel = Skcompute(dataset,PARAMETERS['ML']['MODEL'][1],name=PARAMETERS['NAME']+"_"+PARAMETERS['ML']['MODEL'][0], save=PARAMETERS['SAVE'])
Skmodel.initTraining()
Skmodel.subset_size = PARAMETERS['ML']['SUBSET_SIZE']
Skmodel.split_size = PARAMETERS['ML']['SPLIT_FRAC']
Skmodel.trainModel(
    Xcols=Xcols,
    Ycol=targetCol,
    predictType='Regression',
    randomSeed=PARAMETERS['RANDOM_SEED'],
    comment=""
)
# %%-
