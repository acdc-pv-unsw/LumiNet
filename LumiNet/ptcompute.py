# %%--  Import
import torch
import copy
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import time
import datetime
import gc
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
from scipy import interp
from LumiNet.logger import Logger
from LumiNet.dataset import Dataset
from .matplotlibstyle import *
# %%-

class Ptcompute():
    #----   Class constants

    def __init__(self,datahandler,model,name : str,save : bool):
        ' Initialize attributes and trace files. Default values are saved here'
        #--------   Check if we're using PyTorch
        if not "torch"in str(model.__class__.__bases__): raise ValueError('Passed model not in Torch module')

        #--------   Check for GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.isGPU = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

        #--------   Define files to save
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        tracefile = datahandler.pathDic['traces']+timestamp+"_"+"_trace_"+name+".txt"
        logger = Logger(tracefile)


        #--------   Attributes and default values
        self.name = name
        self.timestamp = timestamp
        self.dh = datahandler
        self.save = save
        self.tracefile = tracefile
        self.split_size = 0.1
        self.logger = logger
        self.model = model
        self.trainNum = 0
        self.subset_size = None
        self.batch_size = None
        self.n_epochs = 25
        self.optimizer = None
        self.loss = None
        self.learning_rate = 0.0001
        self.scaler = None
        self.CM_fz = 13
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
            if k == 'optimizer': continue
            if k == 'loss': continue
            if k == 'learning_rate': continue
            if k == 'n_epochs': continue
            if k == 'batch_size': continue
            if k == 'subset_size': continue
            if k == 'split_size': continue
            if k == 'scaler': continue
            print("\t",k,"-"*(1+len(max(attr,key=len))-len(k)),">",attr[k])
        print("\n")
        self.regResults = []
        self.classResults = []

        #--------   Print model
        title = "MODEL"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        print(self.model)
        print("\n")

        if self.save: self.logger.close()
    def trainModel(self,Ycol,transform,transformTrain=None,randomSeed=None,split_randomSeed=None,comment=""):
        'Main functions that trains the model - MODEL WILL REMEMBER PREVIOUS TRAINING'
        self.trainNum+=1
        if not randomSeed : randomSeed = np.random.randint(1000)
        if not split_randomSeed : split_randomSeed = np.random.randint(1000)
        torch.manual_seed(randomSeed)
        self.comment = comment
        if self.save: self.logger.open()

        title = "TRAINING #"+str(self.trainNum)
        print(">"*60)
        print(" "*np.max([0,np.int((60-len(title))/2)])+title)
        print("<"*60)
        print("\n")

        #--------   Prediction type and normalization if needed
        self.nb_classes=len(self.dh.matchDf[Ycol].value_counts())
        self.predictType = "Classification"
        isReg = False
        if self.nb_classes > 100 :  #   Maximum number of class instances is 100 before switching to regression
            self.predictType = "Regression"
            isReg = True
        if isReg and not self.scaler: self.scaler = preprocessing.MinMaxScaler()
        if isReg: self.scaler.fit(self.dh.matchDf[Ycol].values.reshape(-1,1))

        #--------   Train and Test set splitting
        df = self.dh.matchDf.copy(deep=True)
        if self.subset_size: df = df.sample(self.subset_size, random_state = randomSeed)
        if isReg:   df[Ycol] = self.scaler.transform(df[Ycol].values.reshape(-1,1))
        df_train, df_test = train_test_split(df,test_size = self.split_size,random_state=split_randomSeed)
        partition = {'train': np.array(df_train['path']), 'test': np.array(df_test['path']) }
        labels = {}
        for label, path in zip(df[Ycol],df['path']):
            labels[path]=label
        if not transformTrain : transformTrain=transform
        Train_set = Dataset(partition['train'], labels, transformTrain)
        Test_set = Dataset(partition['test'], labels, transform)

        #--------   Other intiailizations
        if not self.optimizer : self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
        if not self.loss: self.loss = nn.MSELoss() if isReg else nn.CrossEntropyLoss()
        if not self.batch_size: self.batch_size= np.min(np.max(len(df)/100,1),50)   #   Default batch size between 1 and 50 as 1% of dataset

        #--------   Prep saving files
        self.figurefile = self.dh.pathDic['figures']+self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment+".png"
        self.predictfile = self.dh.pathDic['outputs']+self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment+".csv"
        self.modelfile = self.dh.pathDic['models']+self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment+".sav"

        #--------   Log Data information
        title = "HYPERPARAMETERS"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        toprint = {
            "Training ID" : self.trainNum,
            "Random seed":randomSeed,
            "Split random seed":split_randomSeed,
            "Model file" : self.modelfile,
            "Figure file" : self.figurefile,
            "Predicted file" : self.predictfile,
            "Datafile" : self.dh.pathDic['Matchfile'],
            "Dataset length" : len(self.dh.matchDf),
            "Subset requested" : self.subset_size,
            "Training set length" : len(df_train),
            "Testing set length" : len(df_test),
            "Test/train size ratio" : self.split_size,
            "Predicted column" : Ycol,
            "Prediction type" : self.predictType,
            "Number of unique instances" : self.nb_classes,
            "Batch size" : self.batch_size,
            "Learning rate" : self.learning_rate,
            "Number of epochs" : self.n_epochs,
            "Loss function" : self.loss,
            "Optimizer" : self.optimizer,
            "Training set transformation" : transformTrain,
            "Testing set transformation" : transform,
            "Scaler": self.scaler,
            }
        for k in toprint:
            print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
        print("\n")

        #--------   Training loop
        title = "TRAINING"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        if self.isGPU: self.model.cuda()

        for i in range(1):
            #   Get training data
            train_loader = Data.DataLoader(Train_set, batch_size=self.batch_size,shuffle=True,num_workers=4,drop_last=True,)
            test_loader = Data.DataLoader(Test_set,batch_size=self.batch_size,shuffle=False,num_workers=4)
            n_batches = len(train_loader)

            #   Time for printing
            training_start_time = time.time()
            tab_train_loss = []
            tab_test_loss = []

            #   Loop for n_epochs
            self.model.train()
            for epoch in range(self.n_epochs):
                running_loss = 0.0
                print_every = n_batches // 10 if n_batches//10>1 else 1
                start_time = time.time()
                start_epoch_time = time.time()
                tab_epoch_train_loss = []
                print(" ----Epoch {}----".format(epoch+1))
                for i, data in enumerate(train_loader, 0):
                    #   Get inputs
                    inputs, targets = data
                    if self.isGPU: inputs, targets = inputs.cuda(), targets.cuda()

                    #   Set the parameter gradients to zero
                    self.optimizer.zero_grad()

                    #   Forward pass, backward pass, optimize
                    outputs = self.model(inputs)
                    loss_size = self.loss(outputs, targets.float()) if isReg else self.loss(outputs, targets)
                    loss_size.backward()
                    self.optimizer.step()

                    #   Print statistics
                    running_loss += loss_size.item()
                    if self.isGPU:
                        tab_epoch_train_loss.append(loss_size.cpu().detach().numpy())
                    else:
                        tab_epoch_train_loss.append(loss_size.detach().numpy())

                    #Print every 10th batch of an epoch
                    if (i + 1) % (print_every + 1) == 0:
                        print("\t {:d}% \t train loss: {:.2e} took {:.1f}s".format(int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                        #Reset running loss and time
                        running_loss = 0.0
                        start_time = time.time()

                    #   Clear GPU mempry
                    if self.isGPU:
                        inputs, targets, outputs, loss_size = inputs.cpu(), targets.cpu(), outputs.cpu(), loss_size.cpu()
                        del inputs, targets, data, outputs, loss_size
                        torch.cuda.empty_cache()

                tab_train_loss.append(tab_epoch_train_loss)
                #   At the end of the epoch, do a pass on the test set
                self.model.eval()
                total_test_loss = 0
                for i, data in enumerate(test_loader, 0):
                    #   Get inputs
                    inputs, targets = data
                    if self.isGPU: inputs, targets = inputs.cuda(), targets.cuda()

                    #   Forward pass
                    outputs = self.model(inputs)
                    loss_size = self.loss(outputs, targets.float()) if isReg else self.loss(outputs, targets)
                    if self.isGPU:
                        total_test_loss += loss_size.cpu().detach().numpy()
                    else:
                        total_test_loss += loss_size.detach().numpy()

                tab_test_loss.append(1/len(test_loader)*total_test_loss)
                print("\t Done \t test loss : {0:.2e} took {1:.1f}s".format(total_test_loss / len(test_loader),time.time()-start_epoch_time))
                self.model.train()
                if self.isGPU:
                    inputs, targets, outputs, loss_size = inputs.cpu(), targets.cpu(), outputs.cpu(), loss_size.cpu()
                    del inputs, targets, outputs, loss_size
                    torch.cuda.empty_cache()

                print("\n")
        print("\n")

        #--------   Results
        title = "RESULTS"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        totalTrainTime = time.time()-training_start_time
        self.model.eval()

        #   Calculate predcited vs actual values for plot
        tab_Actual=[]
        tab_Pred=[]
        if not isReg :
            tab_Prob=dict()
            for label in sorted(self.dh.matchDf['Bins'].unique()):
                tab_Prob[label]=[]
        test_loader = Data.DataLoader(Test_set,batch_size=1,shuffle=False,num_workers=0)
        for inputs, targets in test_loader:
            if self.isGPU: inputs, targets = inputs.cuda(), targets.cuda()
            #Forward pass
            outputs = self.model(inputs)
            if self.isGPU:
                targets = targets.cpu().detach().numpy()[0]
            else:
                targets = targets.detach().numpy()[0]
            tab_Actual.append(targets)
            if isReg:
                if self.isGPU:
                    outputs = outputs.cpu().detach().numpy().tolist()[0][0]
                else:
                    outputs = outputs.detach().numpy().tolist()[0][0]
                tab_Pred.append(outputs)
            else:
                prediction = nn.Softmax(dim=1)(outputs)
                if self.isGPU:
                    prediction = prediction.cpu().detach().numpy()[0]
                else:
                    prediction = prediction.detach().numpy()[0]
                tab_Pred.append(np.argmax(prediction))
                for i,label in zip(range(self.nb_classes),sorted(self.dh.matchDf['Bins'].unique())):
                    tab_Prob[label].append(prediction[i])
            if self.isGPU:
                inputs=inputs.cpu()
                del inputs, targets, outputs
                torch.cuda.empty_cache()

        #   Report results
        if isReg:
            tab_Actual = self.scaler.inverse_transform(np.array(tab_Actual).reshape(1,-1)).tolist()[0]
            tab_Pred = self.scaler.inverse_transform(np.array(tab_Pred).reshape(1,-1)).tolist()[0]
            slope,intercept,Rsq,_ ,_ = scipy.stats.linregress(tab_Actual, tab_Pred)
            results = {
                "Reference":self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment,
                "Training ID": str(self.trainNum),
                "Total training time (s)": "{:.2f}".format(totalTrainTime),
                "Average training time per epoch (s)": "{:.2f}".format(totalTrainTime/self.n_epochs),
                "Final training score": "{:.2e}".format(np.mean(tab_train_loss[-1])),
                "Best training score": "{:.2e}".format(np.min([np.mean(avg) for avg in tab_train_loss])),
                "Final testing score": "{:.2e}".format(tab_test_loss[-1]),
                "Best testing score": "{:.2e}".format(np.min(tab_test_loss)),
                "True vs predicted slope":"{:.2e}".format(slope),
                "True vs predicted intercept":"{:.2e}".format(intercept),
                "True vs predicted Rsquare":"{:.3f}".format(Rsq),
            }
            for k in results:
                print("\t",k,"-"*(1+len(max(results,key=len))-len(k)),">",results[k])
            self.regResults.append(results)
            print("\n")
        else:
            results = {
                "Reference":self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment,
                "Training ID": str(self.trainNum),
                "Total training time (s)": "{:.2f}".format(totalTrainTime),
                "Average training time per epoch": "{:.2f} s".format(totalTrainTime/self.n_epochs),
                "Final training score": "{:.2e}".format(np.mean(tab_train_loss[-1])),
                "Best training score": "{:.2e}".format(np.min([np.mean(avg) for avg in tab_train_loss])),
                "Final testing score": "{:.2e}".format(tab_test_loss[-1]),
                "Best testing score": "{:.2e}".format(np.min(tab_test_loss)),
                "Weighted Accuracy": "{:.3f}".format(metrics.accuracy_score(tab_Actual,tab_Pred)),
                "Weighted F1-score": "{:.3f}".format(metrics.f1_score(tab_Actual,tab_Pred,average='weighted')),
                "Weighted Precision": "{:.3f}".format(metrics.precision_score(tab_Actual,tab_Pred,average='weighted')),
                "Weighted Recall": "{:.3f}".format(metrics.recall_score(tab_Actual,tab_Pred,average='weighted')),
            }
            #   Compute score per label
            for i,s in zip(range(self.nb_classes),metrics.recall_score(tab_Actual,tab_Pred,average=None)):
                results["Recall - class "+str(i)]="{:.3f}".format(s)
            for i,s in zip(range(self.nb_classes),metrics.precision_score(tab_Actual,tab_Pred,average=None)):
                results["Precision - class "+str(i)]="{:.3f}".format(s)
            for i,s in zip(range(self.nb_classes),metrics.f1_score(tab_Actual,tab_Pred,average=None)):
                results["F1-score - class "+str(i)]="{:.3f}".format(s)
            #   Compute ROC curves
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i,label in zip(range(self.nb_classes),sorted(self.dh.matchDf['Bins'].unique())):
                fpr[i], tpr[i], _ = metrics.roc_curve(tab_Actual, tab_Prob[label],pos_label=i)
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                results["AUC - class "+str(i)] =  "{:.3f}".format(roc_auc[i])
            #   First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.nb_classes)]))

            #   Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.nb_classes):
                mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

            #   Finally average it and compute AUC
            mean_tpr /= self.nb_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
            results["Macro AUC"] = "{:.3f}".format(roc_auc["macro"])


            for k in results:
                print("\t",k,"-"*(1+len(max(results,key=len))-len(k)),">",results[k])

            self.classResults.append(results)
            print("\n")

            title = "LABELS CLASS ID"
            print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
            toprint={}
            for i,label in zip(range(self.nb_classes),sorted(self.dh.matchDf['Bins'].unique())):
                toprint[str(i)]=label
            self.vocab=toprint
            for k in toprint:
                print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
            print("\n")

            title = "CLASSIFICATION REPORT"
            print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
            print(metrics.classification_report(tab_Actual,tab_Pred, digits=3))
            print("\n")

        # Graphs
        for i in range(1):
            plt.figure(figsize = (10,10))
            gs = mpl.gridspec.GridSpec(2, 2)
            tab_epoch = range(1,self.n_epochs+1,1)
            ax1 = plt.subplot(gs[0,:]) if isReg else plt.subplot(gs[0,:-1])
            ax1.set_ylabel('Loss (a.u)', fontsize=14)
            ax1.set_xlabel('Epoch ', fontsize=14)
            ax1.set_title('Learning curves', fontsize=16)
            train_avg_loss=[]
            for x,y in zip(tab_epoch,tab_train_loss):
                #ax1.scatter([x]*len(y),y,c="C1",marker=".")
                train_avg_loss.append(np.mean(y))
            ax1.plot(tab_epoch,train_avg_loss,'.-',c="C1",label='Training loss')
            ax1.plot(tab_epoch,tab_test_loss,'.-',c="C4",label='Testing loss')
            ax1.legend()
            if isReg:
                ax2 = plt.subplot(gs[1,:])
                ax2.set_xlabel('True value', fontsize=14)
                ax2.set_ylabel('Predicted value', fontsize=14)
                ax2.plot([min([min(tab_Actual),min(tab_Pred)]),max([max(tab_Actual),max(tab_Pred)])],[min([min(tab_Actual),min(tab_Pred)]),max([max(tab_Actual),max(tab_Pred)])],linestyle="--",c="C3")
                ax2.scatter(tab_Actual, tab_Pred, c="C0", marker=".")
            else:
                ax2 = plt.subplot(gs[1,:])
                ax2.set_xlabel('Specificity', fontsize=14)
                ax2.set_ylabel('Sensitivity', fontsize=14)
                ax2.set_title('ROC curves', fontsize=16)
                ax2.plot(fpr["macro"],tpr["macro"],label="Macro-average (auc={0:0.3f})".format(roc_auc["macro"]))
                for i,label in zip(range(self.nb_classes),sorted(self.dh.matchDf['Bins'].unique())):
                    plt.plot(fpr[i],tpr[i],linestyle=':',label="{0} (auc={1:0.3f})".format(label,roc_auc[i]))

                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.legend(fontsize=self.CM_fz)

                ax3 = plt.subplot(gs[0,-1])
                CM_labels = sorted(self.dh.matchDf['Bins'].unique())
                CM_data = metrics.confusion_matrix(tab_Actual,tab_Pred)
                try:
                    df_CM =pd.DataFrame(CM_data,index=CM_labels[:len(CM_data)], columns=CM_labels[:len(CM_data)]).transpose()
                except:
                    df_CM =pd.DataFrame(CM_data).transpose()
                self.CM=df_CM
                for i in range(1):
                    sum_CM = np.array( df_CM.to_records(index=False).tolist()).sum()
                    max_CM = np.array( df_CM.to_records(index=False).tolist()).max()
                    self.df_CM =df_CM
                    sns.heatmap(
                        df_CM,
                        annot=True,
                        ax=ax3,
                        cbar=False,
                        fmt="d",
                        square=True,
                        linewidths=.5,
                        linecolor='w',
                        annot_kws={"size": self.CM_fz}
                    )
                    ax3.set_xticklabels(ax3.get_xticklabels(), rotation = 45, fontsize = 10)
                    ax3.set_yticklabels(ax3.get_yticklabels(), rotation = 25, fontsize = 10)
                    for t in ax3.xaxis.get_major_ticks():
                        t.tick1On = False
                        t.tick2On = False
                    for t in ax3.yaxis.get_major_ticks():
                        t.tick1On = False
                        t.tick2On = False
                    for i in range(1):
                        #face colors list
                        quadmesh = ax3.findobj(QuadMesh)[0]
                        facecolors = quadmesh.get_facecolors()

                        #iter in text elements
                        array_df = np.array( df_CM.to_records(index=False).tolist())
                        text_add_glob = []; text_del_glob  = [];
                        posi = -1 #from left to right, bottom to top.
                        fz=self.CM_fz # fontsize of text
                        for oText in ax3.collections[0].axes.texts: #ax.texts:
                            pos = np.array( oText.get_position()) - [0.5,0.5]
                            lin = int(pos[1]); col = int(pos[0]);
                            posi += 1
                            #set text
                            text_add = []; text_del = [];
                            cell_val = array_df[lin][col]
                            tot_all = array_df[-1][-1]
                            per = (float(cell_val) / tot_all) * 100
                            curr_column = array_df[:,col]
                            ccl = len(curr_column)
                            if(per > 0):
                                txt = '%s\n%.2f%%' %(cell_val, per)
                            else:
                                txt = ''

                            oText.set_text(txt)

                            #main diagonal
                            if(col == lin):
                                #set color of the textin the diagonal to white
                                oText.set_color('k')
                                # set background color in the diagonal to blue
                                facecolors[posi] = np.append(np.array(plt.cm.datad['Greens'][np.minimum(5,int(10*cell_val/max_CM))]),1)
                            else:
                                oText.set_color('k')
                                facecolors[posi] = np.append(np.array(plt.cm.datad['Reds'][np.minimum(5,int(10*cell_val/max_CM))]),1)

                            text_add_glob .extend(text_add)
                            text_del_glob .extend(text_del)

                        #remove the old ones
                        for item in text_del_glob:
                            item.remove()
                        #append the new ones
                        for item in text_add_glob:
                            ax3.text(item['x'], item['y'], item['text'], **item['kw'])
                ax3.set_xlabel('True labels', fontsize=14)
                ax3.set_ylabel('Predicted labels', fontsize=14)
                ax3.set_title('Confusion matrix', fontsize=16)

            plt.suptitle(self.timestamp+"_"+self.name+"_"+str(self.trainNum)+"_"+comment, fontsize=18)
            plt.tight_layout(rect=[0,0, 1, 0.93])
            if self.save: plt.savefig(self.figurefile,transparent=True,bbox_inches='tight')
            plt.show()
            plt.close()
        self.lossPlots=pd.DataFrame({'epoch':tab_epoch,'train_loss':train_avg_loss,'test_loss':tab_test_loss})

        #--------   Save model
        if self.save: self.logger.close()
        #   Save self.model
        if self.isGPU:
            self.model=self.model.cpu()
            torch.cuda.empty_cache()
        if self.save: torch.save(self.model.state_dict(),self.modelfile)

        #--------   Save predict file
        df_test['True'] = tab_Actual
        df_test["Predicted"] = tab_Pred
        if self.save: df_test.to_csv(self.predictfile, index = None, header=True)
    def selectCNN(key,nClass,preTrain=True,reqGrad=True):
        #   Adjust models connected layer to nClass
        if "ResNet" in key:
            model = models.resnet18(pretrained=preTrain)
            for param in model.parameters(): param.requires_grad = reqGrad
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.fc.in_features = num_ftrs

        elif "AlexNet" in key:
            model = models.alexnet(pretrained=preTrain)
            for param in model.parameters(): param.requires_grad = reqGrad
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.classifier[6].in_features=num_ftrs
        elif "VGG" in key:
            model = models.vgg11_bn(pretrained=preTrain)
            for param in model.parameters(): param.requires_grad = reqGrad
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.classifier[6].in_features=num_ftrs
        elif "SqueezeNet" in key:
            model = models.squeezenet1_0(pretrained=preTrain)
            for param in model.parameters(): param.requires_grad = reqGrad
            model.classifier[1] = nn.Conv2d(512, nClass, kernel_size=(1,1), stride=(1,1))
            model.num_classes = nClass
        elif "DenseNet" in key:
            model = models.densenet121(pretrained=preTrain)
            for param in model.parameters(): param.requires_grad = reqGrad
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.classifier.in_features = num_ftrs
        else:
            print("No model selected")
            model = None
        return(model)
    def freezeCNN(key,model):
        if "ResNet" in key:
            num_ftrs = model.fc.in_features
            temp_w = model.fc[0].weight.values
            temp_b = model.fc[0].bias.values
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
            )
            model.fc[0].weight.values=temp_w
            model.fc[0].bias.values=temp_b
            model.fc.in_features = num_ftrs
        elif "AlexNet" in key:
            num_ftrs = model.classifier[6].in_features
            temp_w = model.classifier[6][0].weight.values
            temp_b = model.classifier[6][0].bias.values
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
            )
            model.classifier[6][0].weight.values = temp_w
            model.classifier[6][0].bias.values = temp_b
            model.classifier[6].in_features = num_ftrs
        elif "VGG" in key:
            num_ftrs = model.classifier[6].in_features
            temp_w = model.classifier[6][0].weight.values
            temp_b = model.classifier[6][0].bias.values
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
            )
            model.classifier[6][0].weight.values = temp_w
            model.classifier[6][0].bias.values = temp_b
            model.classifier[6].in_features = num_ftrs
        elif "SqueezeNet" in key:
            model.classifier[1] = nn.Dropout(p=0, inplace=False)
        elif "DenseNet" in key:
            num_ftrs = model.classifier.in_features
            temp_w = model.classifier[0].weight.values
            temp_b = model.classifier[0].bias.values
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
            )
            model.classifier[0].weight.values = temp_w
            model.classifier[0].bias.values = temp_b
            model.classifier.in_features = num_ftrs
        else:
            print("No model selected")
            model = None
        return(model)
    def addClassificationCNN(key,model,nClass):
        if "ResNet" in key:
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.fc.in_features = num_ftrs

        elif "AlexNet" in key:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.classifier[6].in_features=num_ftrs
        elif "VGG" in key:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.classifier[6].in_features=num_ftrs
        elif "SqueezeNet" in key:
            model.classifier[1] = nn.Conv2d(512, nClass, kernel_size=(1,1), stride=(1,1))
            model.num_classes = nClass
        elif "DenseNet" in key:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, int(num_ftrs/2)),
                nn.BatchNorm1d(int(num_ftrs/2)),
                nn.ReLU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear( int(num_ftrs/2),nClass)
            )
            model.classifier.in_features = num_ftrs
        else:
            print("No model selected")
            model = None
        return(model)
    def extractFeature(model,df,transform,batch_size=1):
        isGPU = torch.cuda.is_available()
        partition = {'data': np.array(df['path'])}
        labels = {}
        for label, path in zip(df['path'],df['path']):
            labels[path]=label
        set = Dataset(partition['data'], labels, transform)
        loader = Data.DataLoader(set,batch_size=batch_size,shuffle=False,num_workers=4)
        model.eval()
        if isGPU: model.cuda()
        p=0
        print('Extracting features')
        batch_df=pd.DataFrame()
        for inputs, targets in loader:
            p+=1
            print(' \t Batch '+str(p)+' of '+str(len(loader)))
            if isGPU: inputs = inputs.cuda()
            outputs = model(inputs)
            if isGPU:
                outputs = outputs.cpu().detach().numpy().tolist()
                inputs=inputs.cpu()
            np.shape(outputs)
            np.shape(targets)
            batch_df_temp = pd.DataFrame(outputs)
            Xcols = ['CNN_'+str(j) for j in batch_df_temp.columns]
            batch_df_temp.columns = Xcols
            batch_df_temp['path']=targets
            batch_df = batch_df.append(batch_df_temp)
            del inputs, targets, outputs
            torch.cuda.empty_cache()
        if isGPU: model=model.cpu()
        torch.cuda.empty_cache()
        out = pd.merge(df,batch_df,on="path")
        return(Xcols,out)
