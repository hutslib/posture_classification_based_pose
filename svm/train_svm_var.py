#!user/bin/python
# _*_ coding: utf-8 _*_
# -------------------------------------------
#  @description: svm classification using sklearn
#  @author: hts
#  @data: 2020-03-12
#  @version: 1.0
#  @github: hutslib
# -------------------------------------------
# -------------------------------------------
#  @description: add probablity svm
#  @author: hts
#  @data: 2020-03-25
#  @version: 2.0
#  @github: hutslib
# -------------------------------------------
# -------------------------------------------
#  @description: add pro predict 
#  @author: hts
#  @data: 2020-03-31
#  @version: 2.1
#  @github: hutslib
# -------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler  
from sklearn.svm import SVC  
import numpy as np
# from numpy.random import shuffle
import os
import sys
import argparse
class train_svm():

    def __init__(self, keypoints_path, model_save_path, npsave_path, command, model_path):
        
        self.keypoints_folder = keypoints_path
        self.model_save_path = model_save_path
        self.command = command
        self.model_path = model_path
        self.np_path = npsave_path
        files = os.listdir(self.keypoints_folder)
        first_time = True
        #生成特征和标签
        for filename in files:
            print('\033[0;22m filename %s \033[0m' %filename)
            self.keypoints_path = self.keypoints_folder + filename + '/var.txt'
            print(self.keypoints_path)
            self.filename = filename
            self.npsave_path = npsave_path + filename + '/np_var.txt'
            Feature, self.lenx, self.leny = self.feature_generate()
            Label = self.label_generate()
            if first_time == True:
                self.feature = Feature
                self.label = Label
                first_time = False
            else:
                self.feature = np.insert(self.feature, 0, Feature, axis=0)
                self.label = np.insert(self.label, 0, Label, axis=0)
            #print(self.feature[array])
            #print(type(self.feature))
        # print('\022[0;22m save feature.txt in %s\022[0m' %self.np_path)
        np.savetxt(self.np_path+'feature.txt', self.feature, fmt = '%f')
        print('---loading feature done!---')
        print('feature sample size: ', np.shape(self.feature) )
        print()
            #self.label.append(self.label.generate())
        np.savetxt(self.np_path+'label.txt', self.label, fmt = '%d')
        print('---loading label done!---')
        print('label size: ', np.shape(self.label))
        print()
        # #数据归一化
        # self.feature_scaling()
        # np.savetxt('/home/hts/posture_classification_based_pose/svm/feature_scaled.txt', self.feature, fmt = '%f')
        # print('---save feature done!---')
        # np.savetxt('/home/hts/posture_classification_based_pose/svm/label_scale.txt', self.label, fmt = '%f')
        # print()
        # print('---save label done!---')
        # print()
        self.keyboard_control()

    def feature_generate(self): 

        with open(self.keypoints_path, 'r') as file:
            my_data = file.readlines()
            count = 0
            array1 = np.zeros(2,dtype=float)
            array2 = np.zeros((1,2),dtype=float)
            for line in my_data:
                #print(line)
                count += 1
                if count == 2:
                    array1[count-1] = line
                    count = 0
                    #print('ok')
                    #print(array1)
                    array2 = np.insert(array2, 0, array1, axis=0)
                else:
                    # if count%3 != 0:
                        array1[count-1] = line
                    #print(array1[count-1])
            #print(array2)
            array2 = np.delete(array2, -1, axis=0)
            #print(array2)
            axisx = np.shape(array2)[0]
            axisy = np.shape(array2)[1]
            #print(axisx, axisy)
            np.savetxt(self.npsave_path, array2, fmt = '%f')
            # print(axisx, axisy)
            return array2, axisx, axisy
        
    def label_generate(self):
       
        array3 = np.empty((self.lenx, 1))
        for index in range(self.lenx):
            if self.filename == 'a': 
                # print('%d a' %index)
                array3[index] = 1
            if self.filename == 'b':
                # print('%d b' %index)
                array3[index] = 1
            # if self.filename == 'c':
            #     # print('%d c' %index)
            #     array3[index] = 3
            if self.filename == 'd':
                # print('%d d' %index)
                array3[index] = 2
            # if self.filename == 'e':
            #     # print('%d e' %index)
            #     array3[index] = 2
        #print(array3)
        return array3

    def feature_scaling(self):
        sc = StandardScaler()
        self.feature = sc.fit_transform(self.feature)
        self.label = sc.fit_transform(self.label)
        return self.feature, self.label

    def svm_training(self, use_pro = False):

        print('---start training svm---')
        #split dataset in two equal parts
        #print(np.shape(self.feature), np.shape(self.label))
        X_train, X_test, Y_train, Y_test = train_test_split(self.feature, self.label, test_size = 0.5, random_state = 0)
        np.savetxt(self.np_path+'X_train.txt', X_train, fmt = '%f')
        np.savetxt(self.np_path+'X_test.txt', X_test, fmt = '%f')
        np.savetxt(self.np_path+'Y_train.txt', Y_train, fmt = '%d')
        np.savetxt(self.np_path+'Y_test.txt', Y_test, fmt = '%d')
        print('---split data done!---')
        print()
        #set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-3, 3, 7, endpoint = True), 
                             'C': np.logspace(-3, 3, 7, endpoint = True)},
                             {'kernel': ['linear'],'C': np.logspace(-3, 3, 7, endpoint = True)},
                             {'kernel': ['poly'], 'C': np.logspace(-3, 3, 7, endpoint = True), 'degree': [1, 2, 3], 'gamma': np.logspace(-3, 3, 7, endpoint = True)}]
        scores = ['precision', 'recall']

        for score in scores:

            print ('tuning hyper-parameters for %s' % score)
            if use_pro == True:
                clf = GridSearchCV(SVC(probability = True), tuned_parameters, cv = 5, scoring = '%s_weighted' % score)
            else:
                clf = GridSearchCV(SVC(), tuned_parameters, cv = 5, scoring = '%s_weighted' % score)

            print(np.shape(X_train), np.shape(Y_train.ravel()))
            clf.fit(X_train, Y_train.ravel())

        print("Best parameters set found on development set:")  
        print()  
        print(clf.best_params_)  
        print()  
        print("Grid scores on development set:")  
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        # for params, mean_score, scores in clf.cv_results_:  
        #     print("%0.3f (+/-%0.03f) for %r"  
        #         % (mean_score, scores.std() * 2, params))  
        print()  
    
        print("Detailed classification report:")  
        print()  
        print("The model is trained on the full development set.")  
        print("The scores are computed on the full evaluation set.")  
        print()
        target_names = ['lying', 'standing'] 
        if use_pro == True:
            Y_true, Y_pred = Y_test.ravel(), clf.predict_proba(X_test)
            Y_result = []
            np.savetxt(self.np_path+'Y_true.txt', Y_true, fmt = '%d')
            np.savetxt(self.np_path+'Y_pred.txt', Y_pred, fmt = '%f')
            for pre_res in Y_pred:
                # print(pre_res)
                pre_res = pre_res.tolist()
                Y_result.append(pre_res.index(max(pre_res))+1)
                if pre_res.index(max(pre_res)) == 0:
                    print ('lying')
                if pre_res.index(max(pre_res)) == 1:
                    print ('standing')
                # if pre_res.index(max(pre_res)) == 2:
                #     print ('lying on front')
                # if pre_res.index(max(pre_res)) == 3:
                #     print ('sitting')
                # if pre_res.index(max(pre_res)) == 4:
                #     print ('standing')
            # Y_result = Y_pred[:,1]
            print(classification_report(Y_true, Y_result, target_names = target_names))  

            np.savetxt(self.np_path+'Y_result.txt', Y_result, fmt = '%f')

        else:
            Y_true, Y_pred = Y_test.ravel(), clf.predict(X_test)
            print(classification_report(Y_true, Y_pred, target_names = target_names))  
            np.savetxt(self.np_path+'Y_true.txt', Y_true, fmt = '%d')
            np.savetxt(self.np_path+'Y_pred.txt', Y_pred, fmt = '%d')
        # print(classification_report(Y_true, Y_pred, target_names = target_names))  
        print()
        print('SVM model saving ......')
        # model_save_path = '/home/hts/posture_classification_based_pose/svm/train_madel2.m'
        joblib.dump(clf, self.model_save_path)
        
        return 
    
    def predict_pic(self, use_pro = True):

        my_clf = joblib.load(self.model_path)
        print('loading model suc')
        if use_pro == Ture:
            my_pre = my_clf.predict_proba(self.feature)
            for pre_res in my_pre:
                pre_res = pre_res.tolist()
                if pre_res.index(max(pre_res)) == 0:
                    print ('lying')
                if pre_res.index(max(pre_res)) == 1:
                    print ('standing')
                # if pre_res.index(max(pre_res)) == 2:
                #     print ('lying on front')
                # if pre_res.index(max(pre_res)) == 3:
                #     print ('sitting')
                # if pre_res.index(max(pre_res)) == 4:
                #     print ('standing')               
        else:
            my_pre = my_clf.predict(self.feature)
        # target_names = ['lying', 'lie on the side', 'sitting', 'standing']
        #print(my_pre, target_names[my_pre])
        print(my_pre)
        
        return
    
    def predict_video(self, use_pro = True):
        print('predict video start')
        my_clf = joblib.load(self.model_path)
        print('loading model suc')
        print("Detailed classification report:")  
        print()  
        target_names = ['lying', 'standing']
        if use_pro == True:
            Y_true, Y_pred = self.label.ravel(),  my_clf.predict_proba(self.feature)
            # Y_result = Y_pred[:,1]
            print(Y_pred)
            # print(Y_result)
            # print(classification_report(Y_true, Y_result, target_names = target_names))  
            np.savetxt('/home/hts/Desktop/kinect_predict/Y_true.txt', Y_true, fmt = '%d')
            np.savetxt('/home/hts/Desktop/kinect_predict/Y_pred.txt', Y_pred, fmt = '%f')
            # np.savetxt('/home/hts/posture_classification_based_pose/svm/Y_result.txt', Y_result, fmt = '%f')
            Y_result = []
            # np.savetxt(self.np_path+'Y_true.txt', Y_true, fmt = '%d')
            # np.savetxt(self.np_path+'Y_pred.txt', Y_pred, fmt = '%f')
            for pre_res in Y_pred:
                # print(pre_res)
                pre_res = pre_res.tolist()
                Y_result.append(pre_res.index(max(pre_res))+1)
                if pre_res.index(max(pre_res)) == 0:
                    print ('lying')
                if pre_res.index(max(pre_res)) == 1:
                    print ('standing')
                # if pre_res.index(max(pre_res)) == 2:
                #     print ('lying on front')
                # if pre_res.index(max(pre_res)) == 3:
                #     print ('sitting')
                # if pre_res.index(max(pre_res)) == 4:
                #     print ('standing')
            # Y_result = Y_pred[:,1]
            print(classification_report(Y_true, Y_result, target_names = target_names))  

            np.savetxt('/home/hts/Desktop/kinect_predict/Y_result.txt', Y_result, fmt = '%f')

        else:
            Y_true, Y_pred = Y_test.ravel(),  my_clf.predict(X_test)
            print(classification_report(Y_true, Y_pred, target_names = target_names))  
            np.savetxt('/home/hts/Desktop/kinect_predict/Y_true.txt', Y_true, fmt = '%d')
            np.savetxt('/home/hts/Desktop/kinect_predict/Y_pred.txt', Y_pred, fmt = '%d')
            print(classification_report(Y_true, Y_pred, target_names = target_names))
        
        return
  
        

    def keyboard_control(self):

        command = self.command
        try:
            if command == 'predict_pic':
                self.predict_pic()
            elif command == 'svm_training':
                self.svm_training()
            elif command == 'pro_svm_training':
                self.svm_training(use_pro=True)
            elif command == 'predict_video':
                self.predict_video()
            else:
                print("---Invalid Command!---")
        except Exception as e:
            print(e)

# read_data('/home/hts/hts2019/openpose/video_res/Res_keypoints/a/keypoints.txt')
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_path", type = str, default = '/home/hts/Desktop/kinect_test/',
                        help = "path to the folder for the json to be processed (default: None)")
    parser.add_argument("--model_save_path", type = str, default = '/home/hts/Desktop/model/model1.m',
                        help = "path to the folder for the json to be processed (default: None)")
    parser.add_argument("--npsave_path", type = str, default = '/home/hts/Desktop/kinect_nppath/',
                        help = "path to the folder for the results to be saved (default: None)")
    parser.add_argument("--command", type = str, default = 'pro_svm_training',
                        help = "next command(default: 'svm_training')")    
    parser.add_argument("--model_path", type = str, default = '/home/hts/Desktop/model/model1.m',
                        help="path to the model to be loaded (default: None)")    
    # parser.add_argument("--label", type=str, default=None,
    #                     help="label (default: None)")
    args = parser.parse_args()
    keypoints_path = args.keypoints_path
    model_save_path = args.model_save_path
    npsave_path = args.npsave_path
    command = args.command
    model_path = args.model_path
    # label = args.label
    train_svm(keypoints_path, model_save_path, npsave_path, command, model_path)
