import pandas as pd
import pickle
from matplotlib import pyplot as plt
import importlib.util
import keras
import sys
import numpy as np
from sklearn.metrics import *
import seaborn as sns
from __main__ import *
import tensorflow as tf
# import tensorflow_io as tfio
# import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
print('Listing all GPU resources:')
print(tf.config.experimental.list_physical_devices('GPU'))
print()
import tensorflow.keras as keras 
# print(tfp.__version__)
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import pickle
import os
import sys
import importlib.util

class ModelLoader:
    def __init__(self):
        self.model_path = '../model/'
        self.weights_file = self.model_path+'model_weights.h5'
        self.history_file = 'history.pkl'
        
        self.FILTERS = 32
        self.DATA_SIZE = 60000*144*144
        self.BATCH_SIZE = 128
        self.EPOCHS = 100
        self.VERBOSE = 2
        
        self.intermediate_layer_model = None
        self.model = None
        self.calibrator = pickle.load(open(self.model_path+'trained_plattscaling.sav', 'rb'))
        
    def mean_binary_crossentropy(self, y, y_pred):
        return tf.reduce_mean(keras.losses.binary_crossentropy(y, y_pred))


    def dice_coef(self, y, y_pred, axis=(1, 2), smooth=0.01):
        """
        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        prediction = keras.backend.round(y_pred)  # Round to 0 or 1
        intersection = tf.reduce_sum(y * y_pred, axis=axis)
        union = tf.reduce_sum(y + y_pred, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator
        return tf.reduce_mean(coef)
 
    
    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def soft_dice_coef(self, y, y_pred, axis=(1, 2), smooth=0.01):
        """
        Sorenson (Soft) Dice  - Don't round the predictions
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """

        intersection = tf.reduce_sum(y * y_pred, axis=axis)
        union = tf.reduce_sum(y + y_pred, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)
    
    
    def make_model(self):
        '''
        Constructs model and loads in the weights
        '''
        ###No activations applied to last Conv2D layer.
        params_final = dict(kernel_size=(1, 1), padding="same",
                            kernel_initializer="he_uniform")

        params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same", data_format="channels_last",
                      kernel_initializer="he_uniform")

        input_layer = keras.layers.Input(shape=(144, 144, 4), name="input_layer")

        encoder_1_a = keras.layers.Conv2D(self.FILTERS, name='encoder_1_a', **params)(input_layer)
        encoder_1_b = keras.layers.Conv2D(self.FILTERS, name='encoder_1_b', **params)(encoder_1_a)
        batchnorm_1 = keras.layers.BatchNormalization(name='batchnorm_1')(encoder_1_b)
        downsample_1 = keras.layers.MaxPool2D(name='downsample_1')(batchnorm_1)

        encoder_2_a = keras.layers.Conv2D(self.FILTERS*2, name='encoder_2_a', **params)(downsample_1)
        encoder_2_b = keras.layers.Conv2D(self.FILTERS*2, name='encoder_2_b', **params)(encoder_2_a)
        batchnorm_2 = keras.layers.BatchNormalization(name='batchnorm_2')(encoder_2_b)
        downsample_2 = keras.layers.MaxPool2D(name='downsample_2')(batchnorm_2)

        encoder_3_a = keras.layers.Conv2D(self.FILTERS*4, name='encoder_3_a', **params)(downsample_2)
        encoder_3_b = keras.layers.Conv2D(self.FILTERS*4, name='encoder_3_b', **params)(encoder_3_a)
        batchnorm_3 = keras.layers.BatchNormalization(name='batchnorm_3')(encoder_3_b)
        downsample_3 = keras.layers.MaxPool2D(name='downsample_3')(batchnorm_3)

        encoder_4_a = keras.layers.Conv2D(self.FILTERS*8, name='encoder_4_a', **params)(downsample_3)
        encoder_4_b = keras.layers.Conv2D(self.FILTERS*8, name='encoder_4_b', **params)(encoder_4_a)
        batchnorm_4 = keras.layers.BatchNormalization(name='batchnorm_4')(encoder_4_b)
        downsample_4 = keras.layers.MaxPool2D(name='downsample_4')(batchnorm_4)


        encoder_5_a = keras.layers.Conv2D(self.FILTERS*16, name='encoder_5_a', **params)(downsample_4)
        encoder_5_b = keras.layers.Conv2D(self.FILTERS*16, name='encoder_5_b', **params)(encoder_5_a)


        upsample_4 = keras.layers.UpSampling2D(name='upsample_4', size=(2, 2), interpolation="bilinear")(encoder_5_b)
        concat_4 = keras.layers.concatenate([upsample_4, encoder_4_b], name='concat_4')
        decoder_4_a = keras.layers.Conv2D(self.FILTERS*8, name='decoder_4_a', **params)(concat_4)
        decoder_4_b = keras.layers.Conv2D(self.FILTERS*8, name='decoder_4_b', **params)(decoder_4_a)


        upsample_3 = keras.layers.UpSampling2D(name='upsample_3', size=(2, 2), interpolation="bilinear")(decoder_4_b)
        concat_3 = keras.layers.concatenate([upsample_3, encoder_3_b], name='concat_3')
        decoder_3_a = keras.layers.Conv2D(self.FILTERS*4, name='decoder_3_a', **params)(concat_3)
        decoder_3_b = keras.layers.Conv2D(self.FILTERS*4, name='decoder_3_b', **params)(decoder_3_a)


        upsample_2 = keras.layers.UpSampling2D(name='upsample_2', size=(2, 2), interpolation="bilinear")(decoder_3_b)
        concat_2 = keras.layers.concatenate([upsample_2, encoder_2_b], name='concat_2')
        decoder_2_a = keras.layers.Conv2D(self.FILTERS*2, name='decoder_2_a', **params)(concat_2)
        decoder_2_b = keras.layers.Conv2D(self.FILTERS*2, name='decoder_2_b', **params)(decoder_2_a)


        upsample_1 = keras.layers.UpSampling2D(name='upsample_1', size=(2, 2), interpolation="bilinear")(decoder_2_b)
        concat_1 = keras.layers.concatenate([upsample_1, encoder_1_b], name='concat_1')
        decoder_1_a = keras.layers.Conv2D(self.FILTERS, name='decoder_1_a', **params)(concat_1)
        decoder_1_b = keras.layers.Conv2D(self.FILTERS, name='decoder_1_b', **params)(decoder_1_a)

        last_layer = tf.keras.layers.Conv2D(name="last_layer",
                                        filters=1, **params_final)(decoder_1_b)
        output_layer = tf.keras.layers.Activation('sigmoid')(last_layer)

        print()
        print('Input size:', input_layer.shape)
        print('Output size:', output_layer.shape)


        model = keras.models.Model(inputs=input_layer, outputs=output_layer,
                                   name = 'model')

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                      loss=self.dice_coef_loss,
                      metrics=[self.dice_coef_loss, self.mean_binary_crossentropy, "accuracy", self.dice_coef, self.soft_dice_coef],
                      )

        model.load_weights(self.weights_file)
        self.model = model
        model.summary()
        
        layer_name = model.layers[-2].name
        self.intermediate_layer_model = keras.Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        return model
    
    def read_history(self):
        with open(self.model_path+self.history_file, 'rb') as f:
            return pickle.load(f)

    def predict_and_calibrate(self,image_data):
        logits = self.intermediate_layer_model.predict(image_data).flatten()
        calibrated_pred = self.calibrator.predict_proba(logits.reshape(-1,1))
        return calibrated_pred[:,1]
    
class DataLoader:
    def __init__(self):
        self.local_data_path = '../data/'
        self.scratch_path = '/scratch/bdsi_root/bdsi1/snehalbp/'
        self.prediction_path = self.scratch_path + 'predictions/'
        
        self.ptrain = np.load(self.local_data_path+'train_patients.npy').tolist()
        self.ptest = np.load(self.local_data_path+'test_patients.npy').tolist()
        self.pval = np.load(self.local_data_path+'val_patients.npy').tolist()
        
    def get_patient_list(self,partition):
        if partition == 'train':
            return ptrain
        elif partition == 'test':
             return ptest
        elif partition == 'valid':
             return pval
    
    def clinical_information(self, random_sample=False):
        '''
        Returns clinical information of patients in test dataset.
        If random_sample is set True, function will select a patient randomly and return the data.
        '''
        df = pd.read_csv(self.local_data_path+'patient_data.csv',index_col=0)
        df = df.loc[df['Case'].isin(self.ptest)]
        if random_sample:
            return df.sample()
        else:
            return df
        
        
class Patient(DataLoader,ModelLoader):
    def __init__(self, DataLoader, ModelLoader):
        self.dataloader = DataLoader
        self.modelloader = ModelLoader
        self.case = None
        
        self.image_path=None
        self.label_path=None
        self.prediction_path=None
        
        self.image_data=None
        self.label_data=None
        self.prediction_data=None
    
    def create(self,case):
        self.case=case
        self.image_path = self.dataloader.scratch_path+'test_images/{}_image.npy'.format(self.case)
        self.label_path = self.dataloader.scratch_path+'test_labels/{}_label.npy'.format(self.case)
        self.prediction_path = self.dataloader.prediction_path+'{}_prediction.npy'.format(self.case)
    
    def get_information(self):
        df = self.dataloader.clinical_information()
        return df[df['Case']==self.case]
        
    def get_datafiles(self, pred=False):
        self.image_data = np.load(self.image_path)
        self.label_data = np.load(self.label_path)
        if pred:
            self.prediction_data = self.modelloader.predict_and_calibrate(self.image_data).reshape(self.label_data.shape)
        else:
            self.prediction_data = np.load(self.prediction_path)
        return self.image_data, self.label_data, self.prediction_data
    
    def plot_figures(self, z_slice):
        self.image_data, self.label_data, self.prediction_data = self.get_datafiles()
        plt.figure(figsize=(25, 15))

        plt.subplot(1, 6, 1)
        plt.imshow(self.image_data[z_slice,:,:,0], cmap="bone", origin="upper")
        plt.title("FLAIR")


        plt.subplot(1, 6, 2)
        plt.imshow(self.image_data[z_slice,:,:,1], cmap="bone", origin="upper")
        plt.title("T1w")

        
        plt.subplot(1, 6, 3)
        plt.imshow(self.image_data[z_slice,:,:,2], cmap="bone", origin="upper")
        plt.title("t1gd")

        plt.subplot(1, 6, 4)
        plt.imshow(self.image_data[z_slice,:,:,3], cmap="bone", origin="upper")
        plt.title("T2w")

        plt.subplot(1, 6, 5)
        plt.imshow(self.label_data[z_slice,:,:,0], cmap="bone", origin="upper")
        plt.title("Tumor")
        
        plt.subplot(1, 6, 6)
        plt.imshow(self.prediction_data[z_slice,:,:,0], cmap="bone", origin="upper")
        plt.title("Prediction")

        plt.show()
        
class Metrics(Patient):
    def __init__(self, Patient):
        self.patient = Patient
        
        self.threshold = 0.5
        _, self.y_true, self.y_prob = self.patient.get_datafiles()
        self.y_true, self.y_prob = self.y_true.ravel(), self.y_prob.ravel()
        self.y_pred = (self.y_prob > self.threshold).astype(int) 
        
        self.case = self.patient.case
        
        self.TN, self.FP, self.FN, self.TP = 0,0,0,0
        self.tumor_percent = 0
        self.non_tumor_percent = 0
        
    def confusion_matrix(self,plot=False,verbose=False):
        self.TN, self.FP, self.FN, self.TP = confusion_matrix(self.y_true, self.y_pred).ravel()
        total = self.y_pred.shape[0]
        self.tumor_percent = ((self.TP+self.FN)/total)*100
        self.non_tumor_percent = ((self.TN+self.FP)/total)*100
            
        if plot==True:
            clf_flat = np.array([self.TN,self.FP,self.FN,self.TP])
            ax= plt.subplot()

            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ["{0:0.0f}".format(value) for value in
                            clf_flat]
            group_percentages = ["{0:.2%}".format(value) for value in
                                 clf_flat/np.sum(clf_flat)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                      zip(group_names,group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            sns.heatmap(np.reshape(clf_flat,(2,2)), annot=labels, fmt='', cmap='Blues')

            ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
            ax.xaxis.set_ticklabels(['non-tumor', 'tumor'])
            ax.yaxis.set_ticklabels(['non-tumor', 'tumor']);
        
        
        if verbose:
            print("True Negative:",self.TN)
            print("False Positive:",self.FP)
            print("False Negative:",self.FN)
            print("True Positive:",self.TP)
            print("Tumor Percent:",self.tumor_percent)
            print("Non-tumor Percent:",self.non_tumor_percent)
            
        metric = ['Case','TN',
            "FP",
            "FN",
            "TP",
            "tumor_percent",
            "non_tumor_percent"]
        
        value = [self.case, self.TN, self.FP, self.FN, self.TP, self.tumor_percent, self.non_tumor_percent]
            
        return metric, value 
    
    def classical_metrics(self,verbose=True):
        #sensitivity:
        TPR = self.TP/(self.TP+self.FN)
        if verbose: print("Sensitivity:",TPR)

        #sensitivity:
        TNR = self.TN/(self.TN+self.FP)
        if verbose: print("Specificity:",TNR)

        #precision:
        PPV = self.TP/(self.TP+self.FP)
        if verbose: print("Precision:",PPV)

        #negative predictive value (NPV):
        NPV = self.TN/(self.TN+self.FN)
        if verbose: print("Negative predictive value:",NPV)

        #false negative rate/ miss rate:
        FNR = self.FN/(self.FN+self.TP)
        if verbose: print("False negative rate:",FNR)

        #fall-out / false positive rate:
        FPR = self.FP/(self.FP+self.TN)
        if verbose: print("Fall Out:",FPR)

        #false discovery rate:
        FDR = 1 - PPV
        if verbose: print("False discovery rate:",FDR)

        #false omission rate:
        FOR = self.FN/(self.FN+self.TN)
        if verbose: print("False omission rate:",FOR)

        #Threat score / Critical Success Index:
        TS = self.TP/(self.TP+self.FN+self.FP)
        if verbose: print("Threat score:",TS)

        #Acccuracy:
        accuracy = (self.TP + self.TN)/(self.TP+self.TN+self.FP+self.FN)
        if verbose: print("Accuracy:",accuracy)

        #Balanced Accuracy:
        BA = balanced_accuracy_score(self.y_true, self.y_pred)
        if verbose: print("Balanced Accuracy:",BA)

        #F1 Score:
        F1 = (2*self.TP)/(2*self.TP+self.FP+self.FN)
        if verbose: print("F1 / Dice Score:",F1)

        #Matthews Correlation Coefficient:
        # MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        MCC = matthews_corrcoef(self.y_true, self.y_pred)
        if verbose: print("Matthews Correlation Coefficient:",MCC)

        #Informedness / Bookmarker Informedness:
        BM = TPR + TNR - 1
        if verbose: print("Informedness / Bookmarker Informedness",BM)

        #Markedness:
        MK = PPV + NPV - 1
        if verbose: print("Markedness",MK)

        ###COMPILE TOGETHER:
        metric = ['Sensitivity',
                "Specificity",
                "Precision",
                "Negative predictive value",
                "False negative rate",
                "Fall Out",
                "False discovery rate",
                "False omission rate",
                "Threat score",
                "Accuracy",
                "Balanced Accuracy",
                "F1 / Dice Score",
                "Matthews Correlation Coefficient",
                "Informedness / Bookmarker Informedness ",
                "Markedness"]
        value = [TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, accuracy, BA, F1, MCC, BM, MK]

        return metric, value
    
    def similarity_metrics(self,verbose=False):

        jaccard = jaccard_score(self.y_true, self.y_pred)
        if verbose: print("Jaccard Coefficient:",jaccard)
            
        roc_auc = roc_auc_score(self.y_true, self.y_prob)
        if verbose: print("Area under ROC Curve:",roc_auc)
        
        cohen = cohen_kappa_score(self.y_true, self.y_pred)
        if verbose: print("Cohen Kappa:",cohen)
        
        RI = rand_score(self.y_true, self.y_pred)
        if verbose: print("Rand Index:",RI)
        
        MI = mutual_info_score(self.y_true, self.y_pred)
        if verbose: print("Mutual Information:",MI)

        metric = [
        "Jaccard Coefficient",
        "Area under ROC Curve",
        "Cohen Kappa",
        "Rand Index",
        "Mutual Information"]
        value = [jaccard, roc_auc, cohen, RI, MI]
        return metric, value
    
    def calibration_metrics(self, y_true, y_prob, logits, accuracy, confidence):
        '''
        y_true : ground truth labels
        y_prob : predictions of the model between [0,1] {train_pred ; train_pred_calibrated}
        logits: outputs from (n-1) layer of the model
        accuracy,confidence: calibration_curve(...)
        '''
        NLL = log_loss(y_true,y_prob, eps = 1e-7)
        print('NLL : %.4f'%NLL)

        BS = brier_score_loss(y_true, y_prob) 
        print('BS : %.4f'%BS)

        AUROC = roc_auc_score(y_true, y_prob)
        print('AUROC : %.4f'%AUROC)

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        AUCPR = auc(recall, precision)
        print('AUCPR : %.4f'%AUCPR)

        n_bins = len(accuracy) 
        n = len(y_prob) 
        h = np.histogram(a=y_prob, range=(0, 1), bins=n_bins)[0]  
        ece = 0
        for m in np.arange(n_bins):
            ece = ece + (h[m] / n) * np.abs(accuracy[m] - confidence[m])
        print('ECE : %.4f'%ece)
