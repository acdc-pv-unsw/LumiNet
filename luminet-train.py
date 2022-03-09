# %%--  Module Import
#   Package classes
from LumiNet import *
from LumiNet import SaveObj
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
''' ----Note - to do by user
Add NAME
Add DATASET_ID and match to folderpath for Load and select datasets
Add SAVEFOLDER
Datasets folder path have a match.csv file containing TARGET_COL and 'path' columns

'''
# %%--  Parameters
PARAMETERS = {
    "NAME" : "MY NAME",
    "SAVE": True,
    'RANDOM_SEED': np.random.randint(1000),
    "DATASET_ID": [ # Select one or more
        # TO FILL BY USER
    ],
    "TARGET_COL":[ # Select one only
        'Eff',
        # 'Voc_mV',
        # 'Isc_A',
        # 'FF',
    ][0],
    'TARGET_MOD':[ # Select one only
        # 'NA',
        # 'NORM',
        'MMAD',
    ][0],
    'TARGET_MOD_VAL':[20,0.3], # NORM : Mean, Stdev ; MMAD : Median, Mad
    'TARGET_BIN':[ # Select one only
        # '10_QUANTILES',
        # '4_QUANTILES',
        'CUSTOM',
    ][0],
    'TARGET_BIN_CUSTOM':[19+n*0.2 for n in range(int(2/0.2)+1)],
    # 'TARGET_BIN_CUSTOM':[600+n*20 for n in range(int(200/20)+1)]+[2000],
    # 'TARGET_BIN_CUSTOM':[7+n*0.2 for n in range(int(4/0.2)+1)]+[15],
    'TARGET_BIN_CUSTOM_CLIP': True,
    'ML_FRAC':0.1,
    'CALCULATE_STATPARAM':False,
    'CALCULATE_DATASET_STAT':False,
    'SAVEFOLDER' : "MYFOLDER",
    'CNN':{
        'IMAGE_SIZE':224, # Default size for pre-train ImageNet = 224
        'PRETRAIN':True, # Use pre-train weights on ImageNet
        'REQGRAD':True, # Allow further training of pre-train weights
        'SUBSET_SIZE': None, # To use only a subset of the training data
        'BATCH_SIZE': 17, # Training batch size
        'SPLIT_FRAC': 0.05, # Fraction of the training set to use for testing
        'N_EPOCHS': 20, # Number of run epochs
        'CM_FZ': 6, # Font Size in confusion matrix
        'MODEL':[ # Select only one
            # 'ResNet',
            # 'AlexNet',
            'VGG',
            # 'SqueezeNet',
            # 'DenseNet',
        ][0],
    },
    'ML':{
        'MODEL':[ # Select only one
            ('Random Forest',RandomForestRegressor(n_estimators=100, verbose=0,n_jobs = -1)),
            # ("Ada Boost", AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='square')),
            # ("Gradient Boost", GradientBoostingRegressor(verbose=0,loss='ls')),
            # ("Neural Network", MLPRegressor((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')),
            # ("Support Vector Machine", SVR()),
        ][0],
        'SUBSET_SIZE': None, # To use only a subset of the training data
        'SPLIT_FRAC': 0.20, # Fraction of the training set to use for testing
        'LINEAR':[ # Select only one
            # ("Lasso",Lasso()),
            ("Linear Regression",LinearRegression()),
            # ("ElasticNet",ElasticNet()),
            # ("Ridge",Ridge()),
        ][0],
    }

}
PARAMETERS['CNN']['TRANSFORM_AUG']=transforms.Compose([
    transforms.Resize((PARAMETERS['CNN']['IMAGE_SIZE'],PARAMETERS['CNN']['IMAGE_SIZE'])),
    transforms.Grayscale(num_output_channels = 3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
PARAMETERS['CNN']['TRANSFORM']=transforms.Compose([
    transforms.Resize((PARAMETERS['CNN']['IMAGE_SIZE'],PARAMETERS['CNN']['IMAGE_SIZE'])),
    transforms.Grayscale(num_output_channels = 3),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////

# %%--  Initialisation
#  <subcell>    Load and select datasets
All_datasets_location={
    # ADD BY USER
    # 'dataset_id':'path of folder - needs to have match.csv with path column and TARGET_COL'
}
Datasets={}
for dataset_id in PARAMETERS['DATASET_ID']:
    Datasets[dataset_id]=Datahandler(All_datasets_location[dataset_id])
    Datasets[dataset_id].dataset_id = dataset_id

#  </subcell>
#  <subcell>    Extract statistical parameters
if PARAMETERS['CALCULATE_STATPARAM']:
    print("Calculating Statistical Parameters")
    for id in Datasets:
        print("\t Processing... ",id)
        Datasets[id].computeStatParameters(threshold=20)
    print("Done\n")
#  </subcell>
#  <subcell>    Calculate dataset statistical parameters
if PARAMETERS['CALCULATE_DATASET_STAT']:
    print("Calculating Dataset Statistics")
    tab=[]
    IDS = ""
    columns=["Id","Min","Max","Mean","Median","Std","Mad","Min_norm","Max_norm","Mean_norm","Median_norm","Std_norm","Mad_norm","Min_mmad","Max_mmad","Mean_mmad","Median_mmad","Std_mmad","Mad_mmad"]
    for id in Datasets:
        print("\t Processing... ",id)
        IDS+="_"+id
        if PARAMETERS['TARGET_MOD_VAL'] is not None:
            line = Datasets[id].computerDatasetStats(PARAMETERS['TARGET_COL'], fMean = PARAMETERS['TARGET_MOD_VAL'][0], fStd=PARAMETERS['TARGET_MOD_VAL'][1])
        else:
            line = Datasets[id].computeDatasetStats(PARAMETERS['TARGET_COL'])
        tab.append(line)
    dataset_stat = pd.DataFrame(tab)
    dataset_stat.columns=columns
    print(dataset_stat.to_string())
    if PARAMETERS['SAVE']: dataset_stat.to_csv(PARAMETERS['SAVEFOLDER']+"datasetStatistic_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+IDS+"_"+PARAMETERS['TARGET_COL']+".csv", index=False)
    print("Done\n")
#  </subcell>
#  <subcell>    Modify Target
if PARAMETERS['TARGET_MOD']=='NORM':
    print("Modifying target column using 'NORM' approach")
    for id in Datasets:
        print("\t Processing... ",id)
        Datasets[id].normCol(PARAMETERS['TARGET_COL'], fMean = PARAMETERS['TARGET_MOD_VAL'][0], fStd=PARAMETERS['TARGET_MOD_VAL'][1])
        targetCol=PARAMETERS['TARGET_COL']+'_std_norm'
    print("Done\n")
elif PARAMETERS['TARGET_MOD']=='MMAD':
    print("Modifying target column using 'MMAD' approach")
    for id in Datasets:
        print("\t Processing... ",id)
        Datasets[id].mmadCol(PARAMETERS['TARGET_COL'], fMean = PARAMETERS['TARGET_MOD_VAL'][0], fStd=PARAMETERS['TARGET_MOD_VAL'][1])
        targetCol=PARAMETERS['TARGET_COL']+'_std_mmad'
    print("Done\n")
else:
    print("Not modifying target column")
    targetCol=PARAMETERS['TARGET_COL']
#  </subcell>
#  <subcell>    Cell Binning and define nClass
print("Creating Bins for "+targetCol)
for id in Datasets:
    print("Processing... ",id,"\t Binning "+PARAMETERS['TARGET_BIN'])
    df=Datasets[id].matchDf
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
    Datasets[id].addLabels(targetCol,binTab)
print("Done\n")

#  </subcell>
#  <subcell>    Plot targetCol distribution and binning
for id in Datasets:
    Datasets[id].plotCol(targetCol,save=PARAMETERS['SAVE'])
#  </subcell>
# %%-

# %%--  Training
#  <subcell>    Assign Data
print("Assigning dataset for ML and CNN training/validation")
RES={}
RES['TIMESTAMP']=timestamp
ML_df = pd.DataFrame()
CNN_df = pd.DataFrame()
for id in Datasets:
    print("\t Processing... ",id)
    ML_df_temp, CNN_df_temp = Datasets[id].splitData(mlFrac=PARAMETERS['ML_FRAC'],randomSeed=PARAMETERS['RANDOM_SEED'])
    ML_df = ML_df.append(ML_df_temp)
    ML_df=ML_df.sample(frac=1, random_state=PARAMETERS['RANDOM_SEED'])
    CNN_df = CNN_df.append(CNN_df_temp)
    CNN_df=CNN_df.sample(frac=1, random_state=PARAMETERS['RANDOM_SEED'])
print("Done\n")
#  </subcell>
#  <subcell>    Train CNN
RES['CNN']={}
cnnDh=Datahandler(PARAMETERS['SAVEFOLDER']+PARAMETERS['NAME']+"\\")
cnnModel=Ptcompute.selectCNN(PARAMETERS['CNN']['MODEL'],nClass,preTrain=PARAMETERS['CNN']['PRETRAIN'],reqGrad=PARAMETERS['CNN']['REQGRAD'])


cnnDh.matchDf=CNN_df.copy(deep=True)
Ptmodel = Ptcompute(cnnDh,cnnModel,name=PARAMETERS['NAME']+"_"+PARAMETERS['CNN']['MODEL'],save=PARAMETERS['SAVE'])
Ptmodel.timestamp=RES['TIMESTAMP']
Ptmodel.subset_size = PARAMETERS['CNN']['SUBSET_SIZE']
Ptmodel.batch_size = PARAMETERS['CNN']['BATCH_SIZE']
Ptmodel.split_size = PARAMETERS['CNN']['SPLIT_FRAC']
Ptmodel.n_epochs = PARAMETERS['CNN']['N_EPOCHS']
Ptmodel.CM_fz = PARAMETERS['CNN']['CM_FZ']
Ptmodel.initTraining()
Ptmodel.trainModel(
    Ycol="Labels",
    transform=PARAMETERS['CNN']['TRANSFORM'],
    transformTrain=PARAMETERS['CNN']['TRANSFORM_AUG'],
    randomSeed=PARAMETERS['RANDOM_SEED'],
    split_randomSeed=PARAMETERS['RANDOM_SEED'],
    comment=""
    )
RES['CNN']['results'] = Ptmodel.classResults[0]
RES['CNN']['vocab'] = Ptmodel.vocab
RES['CNN']['CM'] = Ptmodel.CM
RES['CNN']['lossPlots']=Ptmodel.lossPlots
RES['CNN']['model_classifier'] = Ptmodel.model
#  </subcell>
#  <subcell>    Extract feature
CNN = Ptcompute.freezeCNN(PARAMETERS['CNN']['MODEL'],Ptmodel.model)
RES['CNN']['model_extractor']=CNN
Xcols, ML_df = Ptcompute.extractFeature(CNN,ML_df,PARAMETERS['CNN']['TRANSFORM'],batch_size=PARAMETERS['CNN']['BATCH_SIZE'])
#  </subcell>
#  <subcell>    Train ML
RES['ML']={}
cnnDh.matchDf=ML_df.copy(deep=True)
Skmodel = Skcompute(cnnDh,PARAMETERS['ML']['MODEL'][1],name=PARAMETERS['NAME']+"_"+PARAMETERS['ML']['MODEL'][0], save=PARAMETERS['SAVE'])
Skmodel.initTraining()
Skmodel.timestamp=RES['TIMESTAMP']
Skmodel.subset_size = PARAMETERS['ML']['SUBSET_SIZE']
Skmodel.split_size = PARAMETERS['ML']['SPLIT_FRAC']
Skmodel.trainModel(
    Xcols=Xcols,
    Ycol=targetCol,
    predictType='Regression',
    randomSeed=PARAMETERS['RANDOM_SEED'],
    comment=""
)
RES['ML']['results']=Skmodel.regResults[0]
RES['ML']['model']=Skmodel.model
#  </subcell>
#  <subcell>    Train Linear
statparam_col=['mu', 'ICA', 'kur', 'skew', 'en', 'sp', 'fw', 'md', 'sd', 'kstat', 'var', 'mu_img', 'med_img', 'sum_img','sd_img']
if PARAMETERS['CALCULATE_STATPARAM']:
    cnnDh.matchDf=ML_df.copy(deep=True)
    Skmodel = Skcompute(cnnDh,PARAMETERS['ML']['LINEAR'][1],name=PARAMETERS['NAME']+"_"+PARAMETERS['ML']['LINEAR'][0], save=PARAMETERS['SAVE'])
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
    RES['ML']['linear_results']=Skmodel.regResults[0]
    RES['ML']['linear_model']=Skmodel.model
#  </subcell>
#  <subcell>    Save model
RES['PARAMETERS']=PARAMETERS
if PARAMETERS['SAVE']: SaveObj(RES,PARAMETERS['SAVEFOLDER']+PARAMETERS['NAME']+"\\",timestamp+"_results")
#  </subcell>
# %%-
