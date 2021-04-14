import os
import pandas as pd
import cv2
import scipy.stats as stat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from .matplotlibstyle import *
import datetime

class Datahandler():
    'Matches EL images paths to IV data based on imput columns'

    def __init__(self,workdir,ELfolderpath=None,IVfile=None):
        'initialize and create folder'
        self.dataset_id = None
        #   Create directory for computation on this dataset
        self.pathDic = {
            'workdir':      workdir,
            'ELfolderpath': ELfolderpath,
            'IVfile':       IVfile,
            'figures':      workdir+"figures\\",
            'models':       workdir+"models\\",
            'traces':       workdir+"traces\\",
            'outputs':      workdir+"outputs\\",
            'Matchfile':    workdir+"match.csv",
        }
        for key, value in self.pathDic.items():
            if key in ['ELfolderpath','IVfile','Matchfile']: continue
            if not os.path.exists(value):   os.mkdir(value)
        if os.path.exists(self.pathDic['Matchfile']):
            self.loadMatchData()
    def readEL(self):
        'Read images from ELfolderpath and store in dataframe'
        if not self.pathDic['ELfolderpath']: raise ValueError('ELfolderpath not defined')
        images = []
        for subdir,dirs,files in os.walk(self.pathDic['ELfolderpath']):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext == ".db": continue
                name = os.path.splitext(file)[0]
                size = os.path.getsize(subdir+"\\"+file)
                location = subdir+"\\"+file
                line = size,ext,name,location
                images.append(line)
        self.ELdf = pd.DataFrame(images)
        self.ELdf.columns=['size','extension','filename','path']
    def readIV(self,sep=","):
        'Read IV data from IVfile csv'
        if not self.pathDic['IVfile']: raise ValueError('IVfile not defined')
        self.IVdf = pd.read_csv(self.pathDic['IVfile'], sep=sep)
    def matchData(self,matchIVcol, matchELcol,keepcol=None):
        'Join both EL and IV dataframe and save it in Matchfile as a csv'
        #   Inner join of both dataframes
        self.matchDf= pd.merge(self.ELdf,self.IVdf,left_on=matchELcol,right_on=matchIVcol, how='inner')
        self.matchDf.fillna(0, inplace=True)
        if keepcol: self.matchDf = self.matchDf[keepcol]
        self.matchDf.to_csv(self.pathDic['Matchfile'],encoding='utf-8', index=False)
    def loadMatchData(self):
        'Load the data if available'
        if os.path.exists(self.pathDic['Matchfile']):
            self.matchDf = pd.read_csv(self.pathDic['Matchfile'])
            self.matchDf.fillna(0, inplace=True)
    def computeStatParameters(self,threshold=20):
        'Load images stored in path and compute the EL parameters'
        stat_feature = {'mu':[],'ICA':[],'kur':[],'skew':[],'en':[],'sp':[],'fw':[], 'md':[], 'sd':[], 'kstat':[], 'var':[], 'mu_img':[], 'med_img':[], 'sum_img':[], 'sd_img':[]}
        for file in self.matchDf['path'].values:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            hist2,bins=np.histogram(img,256,[0,256])
            threshold=min(threshold,len(hist2)-1)
            hist = [h if b>threshold else 0 for b,h in zip(bins,hist2)]
            PEL=hist/np.sum(hist)
            np.sum(PEL)
            len(hist)
            stat_feature['mu'].append(np.mean(hist))
            stat_feature['md'].append(np.median(PEL))
            stat_feature['ICA'].append(100/np.sum(hist2)*np.sum([hist2[i] for i in range(threshold+1)]))
            stat_feature['sd'].append(np.std(PEL))
            stat_feature['kur'].append(stat.kurtosis(PEL))
            stat_feature['skew'].append(stat.skew(PEL))
            stat_feature['kstat'].append(stat.kstat(PEL))
            stat_feature['var'].append(stat.variation(PEL))
            stat_feature['en'].append(stat.entropy(PEL))
            stat_feature['sp'].append(np.ptp(PEL))
            stat_feature['fw'].append(((5/100)*(np.max(PEL)))-((5/100)*(np.min(PEL))))
            stat_feature['mu_img'].append(np.mean(img))
            stat_feature['med_img'].append(np.median(img))
            stat_feature['sum_img'].append(np.sum(img))
            stat_feature['sd_img'].append(np.std(img))
        for key,value in stat_feature.items():
            self.matchDf[key]=value
        self.matchDf.fillna(0, inplace=True)
        self.matchDf.to_csv(self.pathDic['Matchfile'],encoding='utf-8', index=False)
    def computerDatasetStats(self,targetCol,fMean=20, fStd=0.3):
        'Compute the normalized and mmad statistics of the target column'
        self.normCol(targetCol=targetCol,fMean=fMean,fStd=fStd)
        self.mmadCol(targetCol=targetCol,fMean=fMean,fStd=fStd)
        self.matchDf[targetCol+'_std_norm']=[eff*fStd+fMean for eff in self.matchDf[targetCol+'_norm']]
        self.matchDf[targetCol+'_std_mmad']=[eff*fStd+fMean for eff in self.matchDf[targetCol+'_mmad']]
        df = self.matchDf
        line=[
            self.dataset_id,
            df[targetCol].min(),
            df[targetCol].max(),
            df[targetCol].mean(),
            df[targetCol].median(),
            df[targetCol].std(),
            df[targetCol].mad(),
            df[targetCol+'_std_norm'].min(),
            df[targetCol+'_std_norm'].max(),
            df[targetCol+'_std_norm'].mean(),
            df[targetCol+'_std_norm'].median(),
            df[targetCol+'_std_norm'].std(),
            df[targetCol+'_std_norm'].mad(),
            df[targetCol+'_std_mmad'].min(),
            df[targetCol+'_std_mmad'].max(),
            df[targetCol+'_std_mmad'].mean(),
            df[targetCol+'_std_mmad'].median(),
            df[targetCol+'_std_mmad'].std(),
            df[targetCol+'_std_mmad'].mad(),
        ]
        return line
    def addLabels(self,binCol,binTab):
        'Bins match dataset binCol column based on binTab and change matchDf in place'
        self.binCol = binCol
        self.binTab = binTab
        ohe = OneHotEncoder(sparse=False,categories='auto')
        self.matchDf['Bins'] = pd.cut(self.matchDf[binCol],binTab,include_lowest=True)
        self.matchDf['Labels'],self.binLevels = pd.factorize(self.matchDf['Bins'],sort=True)
        self.matchDf['OHE']=[elt for elt in ohe.fit_transform(self.matchDf['Labels'].values.reshape(-1,1))]
        self.ohe = ohe
        self.matchDf.to_csv(self.pathDic['Matchfile'],encoding='utf-8', index=False)
    def saveMatchDf(self):
        self.matchDf.to_csv(self.pathDic['Matchfile'],encoding='utf-8', index=False)
    def normCol(self,targetCol, fMean, fStd):
        'Normalize with Mean and Standard deviation and bring to fMean and fStd'
        mean, std = self.matchDf[targetCol].mean(), self.matchDf[targetCol].std()
        self.matchDf[targetCol+"_norm"] = [(eff-mean)/std for eff in self.matchDf[targetCol]]
        self.matchDf[targetCol+'_std_norm']=[eff*fStd+fMean for eff in self.matchDf[targetCol+'_norm']]
        self.saveMatchDf()
    def mmadCol(self,targetCol, fMean, fStd):
        'Normalize with Median and Mean Absolute Deviation and bring to fMean and fStd'
        med, mad = self.matchDf[targetCol].median(), self.matchDf[targetCol].mad()
        self.matchDf[targetCol+"_mmad"] = [(eff-med)/mad for eff in self.matchDf[targetCol]]
        self.matchDf[targetCol+'_std_mmad']=[eff*fStd+fMean for eff in self.matchDf[targetCol+'_mmad']]
        self.saveMatchDf()
    def plotCol(self,targetCol, save=False):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (8,8))
        df=self.matchDf
        hist_level = np.power(10,np.floor(np.log10(df[targetCol].mad()))-1)
        bins = np.arange(df[targetCol].mean()-3*df[targetCol].std(),df[targetCol].mean()+3.1*df[targetCol].std(),hist_level)
        df[targetCol].hist(ax = axes, bins = bins)
        axes.annotate("Mean: \t %.1F  \nStdev: \t %.2F "%(df[targetCol].mean(),df[targetCol].std()),xy=(0.05,0.9),xycoords='axes fraction', fontsize=14)
        axes.set_xlabel("%s"%(targetCol), fontsize = 14)
        axes.set_ylabel("Counts [per %s]"%(hist_level), fontsize = 14)
        plt.xlim([np.floor(self.binTab[1]-df[targetCol].std()),np.ceil(self.binTab[-1]+df[targetCol].std())])
        plt.title("%s ; Dataset size: %s"%(self.dataset_id,len(df)), fontsize=18)
        for l,c,b in zip(range(len(axes.patches)),df['Bins'].value_counts(sort=False),self.binLevels):
            axes.annotate("("+str(l)+") "+str(b)+": "+"\t {:.1f}".format(100*c/len(df))+"%",xy=(0.05,0.86-0.04*l),xycoords='axes fraction', fontsize=10)
        plt.tight_layout()
        if save: plt.savefig(self.pathDic['figures']+"Distribution_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_"+targetCol+".png",transparent=True,bbox_inches='tight')
        plt.show()
    def splitData(self,mlFrac, randomSeed = None):
        All_df = self.matchDf.copy(deep=True)
        Outlier_df = All_df.loc[All_df['Labels']==0]
        Outlier_df=Outlier_df.sample(frac=1, random_state=randomSeed) #Shuffle
        Normal_df = All_df.loc[All_df['Labels']!=0]
        Normal_df=Normal_df.sample(frac=1, random_state=randomSeed) #Shuffle

        ML_df , CNN_df = train_test_split(Normal_df, train_size=mlFrac, test_size=1-mlFrac, random_state=randomSeed)
        CNN_df = CNN_df.append(Outlier_df)

        #Shuffle all
        ML_df=ML_df.sample(frac=1, random_state=randomSeed)
        CNN_df=CNN_df.sample(frac=1, random_state=randomSeed)
        return ML_df, CNN_df
