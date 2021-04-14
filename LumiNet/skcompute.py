# %%-- Imports
from LumiNet.logger import Logger
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import datetime
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from .matplotlibstyle import *
# %%-
class Skcompute():
    #----   Class constants

    def __init__(self, datahandler, model, name : str, save : bool):
        ' Initialize attributes and trace files. Default values are saved here'
        #--------   Check if we're using PyTorch
        if not "sklearn"in str(model.__class__.__bases__): raise ValueError('Passed model not in Sklearn module')

        #--------   Define files to save
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        tracefile = datahandler.pathDic['traces']+timestamp+"_"+"_trace_"+name+".txt"
        logger = Logger(tracefile)

        #--------   AttributesSk
        self.name = name
        self.timestamp = timestamp
        self.model = model
        self.dh = datahandler
        self.save = save
        self.tracefile = tracefile
        self.logger = logger
        self.trainNum = 0
        self.subset_size = None
        self.randomSeedModel = None
        self.split_size = 0.1

    def initTraining(self):
        'Record Hyper parameter on console and log file'

        if self.save: self.logger.open()

        #--------   Print name
        print(">"*80)
        print(" "*np.max([0,np.int((80-len(self.name))/2)])+self.name)
        print("<"*80)
        print("\n")

        #--------   Print attributes
        title = "ATTRIBUTES"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        attr = self.__dict__
        attr['Working directory']=self.dh.pathDic['workdir']
        for k in attr:
            if k == "model" : continue
            if k == "logger" : continue
            if k == "trainNum" : continue
            if k == 'dh': continue
            if k == 'split_size': continue
            print("\t",k,"-"*(1+len(max(attr,key=len))-len(k)),">",attr[k])
        print("\n")
        self.regResults = []
        self.classResults = []

        #--------   Print model
        title = "MODEL"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        try:
            if self.randomSeedModel:
                self.model.random_state = np.randomSeedModel
            else:
                self.model.random_state = np.random.randint(1000)
        except:
            pass
        attr = self.model.get_params()
        for p in attr:
            print("\t",p,"-"*(1+len(max(attr,key=len))-len(p)),">",attr[p])
        print("\n")

        if self.save: self.logger.close()

    def trainModel(self,Xcols,Ycol,predictType,randomSeed=None,comment=""):
        'Main functions that trains the model'
        self.trainNum+=1
        if not randomSeed : randomSeed = np.random.randint(1000)
        self.predictType=predictType
        self.comment = comment
        if self.save: self.logger.open()

        title = "TRAINING #"+str(self.trainNum)
        print(">"*60)
        print(" "*np.max([0,np.int((60-len(title))/2)])+title)
        print("<"*60)
        print("\n")

        #--------   Train and Test set splitting
        if self.subset_size:
            df = self.dh.matchDf.sample(self.subset_size, random_state = randomSeed)
        else:
            df = self.dh.matchDf
        df_train, df_test = train_test_split(df,test_size = self.split_size,random_state=randomSeed)
        X = df[Xcols]
        Y = df[Ycol]
        X_train, X_test, Y_train, Y_test= df_train[Xcols],df_test[Xcols],df_train[Ycol],df_test[Ycol]
        #--------   Prep saving files
        self.figurefile = self.dh.pathDic['figures']+self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment+".png"
        self.predictfile = self.dh.pathDic['outputs']+self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment+".csv"
        self.modelfile = self.dh.pathDic['models']+self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment+".sav"

        #--------   Log Data information
        title = "HYPERPARAMETERS"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        toprint = {
            "Training ID" : self.trainNum,
            "Model file" : self.modelfile,
            "Figure file" : self.figurefile,
            "Predicted file" : self.predictfile,
            "Datafile" : self.dh.pathDic['Matchfile'],
            "Dataset length" : len(X),
            "Subset requested" : self.subset_size,
            "Training set length" : len(X_train),
            "Testing set length" : len(X_test),
            "Feature column(s)" : Xcols,
            "Predicted column" : Ycol,
            "Prediction type" : predictType,
            "Training random seed":randomSeed,
            "Test/train size ratio" : self.split_size,
            }
        for k in toprint:
            print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
        print("\n")

        #--------   Train model and print verbose
        title = "TRAINING VERBOSE"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        training_start_time = time.time()
        self.model.fit(X_train,Y_train)
        training_end_time = time.time()
        print("\n")

        #--------   Print Results summary amd save model
        title = "RESULTS"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        if predictType == "Regression":
            results = {
                "Reference":self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment,
                "Training ID": str(self.trainNum),
                "Training time (s)": "{:.2f}".format(training_end_time-training_start_time),
                "Train set score": "{:.3f}".format(self.model.score(X_train,Y_train)),
                "Test set score": "{:.3f}".format(self.model.score(X_test,Y_test)),
                "Train set RMSE":"{:.3e}".format(np.sqrt(metrics.mean_squared_error(Y_train,self.model.predict(X_train)))),
                "Test set RMSE":"{:.3e}".format(np.sqrt(metrics.mean_squared_error(Y_test,self.model.predict(X_test)))),
            }
            self.regResults.append(results)
            for k in results:
                print("\t",k,"-"*(1+len(max(results,key=len))-len(k)),">",results[k])
            print("\n")

        if predictType == "Classification":
            results = {
                "Reference":self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment,
                "Training ID": str(self.trainNum),
                "Training time (s)": "{:.2f}".format(training_end_time-training_start_time),
                "Train set score": "{:.3f}".format(self.model.score(X_train,Y_train)),
                "Test set score": "{:.3f}".format(self.model.score(X_test,Y_test)),
                "Accuracy": "{:.3f}".format(metrics.accuracy_score(Y_test,self.model.predict(X_test))),
                "F1-score": "{:.3f}".format(metrics.f1_score(Y_test,self.model.predict(X_test),average='micro')),
                "Precision": "{:.3f}".format(metrics.precision_score(Y_test,self.model.predict(X_test),average='micro')),
                "Recall": "{:.3f}".format(metrics.recall_score(Y_test,self.model.predict(X_test),average='micro')),
                #"Area under ROC curve": "{:.3f}".format(metrics.roc_auc_score(Y_test,self.model.predict_proba(X_test)[:,0])),
            }
            self.classResults.append(results)
            for k in results:
                print("\t",k,"-"*(1+len(max(results,key=len))-len(k)),">",results[k])
            print("\n")

            title = "CLASSIFICATION REPORT"
            print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
            print(metrics.classification_report(Y_test,self.model.predict(X_test), digits=3))

        if self.save: self.logger.close()
        if self.save: joblib.dump(self.model,self.modelfile)

        #--------   Plot graph
        if predictType == "Regression":
            Y_pred =self.model.predict(X_test)
        if predictType == "Classification":
            Y_pred = self.model.predict(X_test)
            Y_pred_proba= [max(p) for p in self.model.predict_proba(X_test)]

        if predictType == "Regression":
            for i in range(1):
                plt.close()
                plt.figure(figsize=(6,6))
                plt.title(self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment)
                ax1 = plt.gca()
                ax1.set_xlabel('Actual value')
                ax1.set_ylabel('Predicted value')
                ax1.scatter(Y_test, Y_pred, c="C0", marker=".", label=self.name+" -#"+str(self.trainNum))
                ax1.annotate(r"R$^2$=%.3F"%(self.model.score(X_test,Y_test)), xy=(0.05,0.90), xycoords='axes fraction')
                ax1.plot([np.min([np.min(Y_test),np.min(Y_pred)]),np.max([np.max(Y_test),np.max(Y_pred)])],[np.min([np.min(Y_test),np.min(Y_pred)]),np.max([np.max(Y_test),np.max(Y_pred)])], linewidth=1 ,linestyle="--",c="C3", label="y=x")
                if self.save: plt.savefig(self.figurefile,transparent=True,bbox_inches='tight')
                plt.show()
                plt.close()

        if predictType == "Classification":
            df_CM =pd.DataFrame(metrics.confusion_matrix(Y_test,self.model.predict(X_test)), columns=np.unique(Y))
            df_CM.index = np.unique(Y)
            cbar_ticks = [np.power(10, i) for i in range(0,2+int(np.log10(len(Y_test))))]
            for i in range(1):
                plt.close()
                plt.figure(figsize=(10,10))
                plt.title(self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment)
                ax1 = plt.gca()
                sns.heatmap(
                    df_CM,
                    annot=True,
                    ax=ax1,
                    cmap=plt.cm.viridis,
                    fmt="d",
                    square=True,
                    linewidths=.5,
                    linecolor='k',
                    vmin=0.9,
                    vmax=1+np.power(10,(1+int(np.log10(len(Y_test))))),
                    norm=mpl.colors.LogNorm(vmin=0.9,vmax=1+np.power(10,(1+int(np.log10(len(Y_test)))))),
                    cbar_kws={'ticks':cbar_ticks, 'orientation':'horizontal'},
                )
                ax1.set_xlabel('Predicted labels', fontsize=14)
                ax1.set_ylabel('Actual labels', fontsize=14)
                if self.save: plt.savefig(self.figurefile,transparent=True,bbox_inches='tight')
                plt.show()
                plt.close()

        #--------   Save predict file
        df_test["True"] = df_test[Ycol]
        df_test["Predicted"] = Y_pred
        if self.save: df_test.to_csv(self.predictfile, index = None, header=True)
