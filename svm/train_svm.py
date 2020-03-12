from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import classification_report  
from sklearn.svm import SVC  
import numpy as np
from numpy.random import shuffle
import os
import sys
import argparse
class train_svm():

    def __init__(self, keypoints_path, npsave_path, command, model_path):
        
        self.keypoints_folder = keypoints_path
        self.command = command
        self.model_path = model_path
        files = os.listdir(self.keypoints_folder)
        first_time = True
        for filename in files:
            print(filename)
            self.keypoints_path = self.keypoints_folder + filename + '/keypoints.txt'
            print(self.keypoints_path)
            self.filename = filename
            self.npsave_path = self.keypoints_folder + filename + '/np_keypoints.txt'
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
        np.savetxt('/home/hts/hts2019/openpose/svm/feature.txt', self.feature, fmt = '%f')
        print('---loading feature done!---')
        print('feature sample size', np.shape(self.feature) )
        print()
            #self.label.append(self.label.generate())
        np.savetxt('/home/hts/hts2019/openpose/svm/label.txt', self.label, fmt = '%f')
        print('---loading label done!---')
        print('label size', np.shape(self.label))
        print()       
        self.keyboard_control()

    def feature_generate(self):

        with open(self.keypoints_path, 'r') as file:
            my_data = file.readlines()
            count = 0
            array1 = np.zeros(33)
            array2 = np.zeros((1,33))
            for line in my_data:
                #print(line)
                count += 1
                if count == 34:
                    count = 0
                    #print('ok')
                    #print(array1)
                    array2 = np.insert(array2, 0, array1, axis=0)
                else:
                    array1[count-1] = line
                    #print(array1[count-1])
            #print(array2)
            array2 = np.delete(array2, -1, axis=0)
            #print(array2)
            axisx = np.shape(array2)[0]
            axisy = np.shape(array2)[1]
            #print(axisx, axisy)
            np.savetxt(self.npsave_path, array2, fmt = '%d')
            return array2, axisx, axisy
        
    def label_generate(self):
       
        array3 = np.empty((self.lenx, 1))
        for index in range(self.lenx):
            if self.filename == 'a':
                array3[index] = 1
            if self.filename == 'b':
                array3[index] = 2
            if self.filename == 'c':
                array3[index] = 3
            if self.filename == 'd':
                array3[index] = 4
            if self.filename == 'e':
                array3[index] = 5
        #print(array3)
        return array3

    def svm_training(self):

        print('---start training svm---')
        #split dataset in two equal parts
        #print(np.shape(self.feature), np.shape(self.label))
        X_train, X_test, Y_train, Y_test = train_test_split(self.feature, self.label, test_size = 0.5, random_state = 0)
        np.savetxt('/home/hts/hts2019/openpose/svm/X_train.txt', X_train, fmt = '%f')
        np.savetxt('/home/hts/hts2019/openpose/svm/X_test.txt', X_test, fmt = '%f')
        np.savetxt('/home/hts/hts2019/openpose/svm/Y_train.txt', Y_train, fmt = '%d')
        np.savetxt('/home/hts/hts2019/openpose/svm/Y_test.txt', Y_test, fmt = '%d')
        print('---split data done!---')
        print()
        #set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-4, 4, 9, endpoint = True), 
                             'C': np.logspace(-4, 4, 9, endpoint = True)},
                             {'kernel': ['linear'],'C': np.logspace(-4, 4, 9, endpoint = True)},
                             {'kernel': ['poly'], 'C': np.logspace(-4, 4, 9, endpoint = True), 'degree': [1, 2, 3], 'gamma': np.logspace(-4, 4, 9, endpoint = True)}]
        scores = ['precision', 'recall']

        for score in scores:

            print ('tuning hyper-parameters for %s' % score)
            clf = GridSearchCV(SVC(decision_function_shape='ovr'), tuned_parameters, cv = 5, scoring = '%s_weighted' % score)
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
        target_names = ['lying', 'lie on the side', 'sitting', 'standing'] 
        Y_true, Y_pred = Y_test.ravel(), clf.predict(X_test)
        np.savetxt('/home/hts/hts2019/openpose/svm/Y_true.txt', Y_true, fmt = '%d')
        np.savetxt('/home/hts/hts2019/openpose/svm/Y_pred.txt', Y_pred, fmt = '%d')
        print(classification_report(Y_true, Y_pred, target_names = target_names))  
        print()

        print('SVM model saving ......')
        model_save_path = '/home/hts/hts2019/openpose/svm/train_madel1.m'
        joblib.dump(clf, model_save_path)
        
        return 
    
    def predict_pic(self):

        my_clf = joblib.load(self.model_path)
        my_pre = my_clf.predict(self.feature)
        target_names = ['lying', 'lie on the side', 'sitting', 'standing']
        #print(my_pre, target_names[my_pre])
        print(my_pre)
        
        return
    
    def predict_video(self):

        my_clf = joblib.load(self.model_path)
        print("Detailed classification report:")  
        print()  
        target_names = ['lying', 'lie on the side', 'sitting', 'standing'] 
        Y_true, Y_pred = self.label.ravel(), my_clf.predict(self.feature)
        np.savetxt('/home/hts/hts2019/openpose/svm/Y_true.txt', Y_true, fmt = '%d')
        np.savetxt('/home/hts/hts2019/openpose/svm/Y_pred.txt', Y_pred, fmt = '%d')
        print(classification_report(Y_true, Y_pred, target_names = target_names))
        
        return
  
        

    def keyboard_control(self):

        command = self.command
        try:
            if command == 'predict_pic':
                self.predict_pic()
            elif command == 'svm_training':
                self.svm_training()
            elif command == 'predict_video':
                self.predict_video()
            else:
                print("---Invalid Command!---")
        except Exception as e:
            print(e)

# read_data('/home/hts/hts2019/openpose/video_res/Res_keypoints/a/keypoints.txt')
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_path", type = str, default = None,
                        help = "path to the folder for the json to be processed (default: None)")
    parser.add_argument("--npsave_path", type = str, default = None,
                        help = "path to the folder for the results to be saved (default: None)")
    parser.add_argument("--command", type = str, default = 'svm_training',
                        help = "next command(default: 'svm_training')")    
    parser.add_argument("--model_path", type = str, default = None,
                        help="path to the model to be loaded (default: None)")    
    # parser.add_argument("--label", type=str, default=None,
    #                     help="label (default: None)")
    args = parser.parse_args()
    keypoints_path = args.keypoints_path
    npsave_path = args.npsave_path
    command = args.command
    model_path = args.model_path
    # label = args.label
    train_svm(keypoints_path, npsave_path, command, model_path)
