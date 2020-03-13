#!user/bin/python
# _*_ coding: utf-8 _*_

# -------------------------------------------
#  @description:  decision tree using sklearn
#  @author: hts
#  @data: 2020-03-13
#  @version: 1.0
#  @github: hutslib
# -------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import numpy as np
import os
import sys
import argparse
import pydotplus

class train_decision_tree():

    def __init__(self, keypoints_path, npsave_path, command, model_path):
        
        self.keypoints_folder = keypoints_path
        self.command = command
        self.model_path = model_path
        files = os.listdir(self.keypoints_folder)
        first_time = True
        #生成特征和标签
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
            #print(type(self.feature))train_test_split) )
        print()
            #self.label.append(self.label.generate())
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/label.txt', self.label, fmt = '%d')
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
            np.savetxt(self.npsave_path, array2, fmt = '%f')
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

    def feature_scaling(self):
        sc = StandardScaler()
        self.feature = sc.fit_transform(self.feature)
        self.label = sc.fit_transform(self.label)
        return self.feature, self.label

    def decision_tree_training(self):

        print('---start training decision tree---')
        #split dataset in two equal parts
        #print(np.shape(self.feature), np.shape(self.label))
        X_train, X_test, Y_train, Y_test = train_test_split(self.feature, self.label, test_size = 0.25, random_state = 0)
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/X_train.txt', X_train, fmt = '%f')
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/X_test.txt', X_test, fmt = '%f')
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/Y_train.txt', Y_train, fmt = '%d')
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/Y_test.txt', Y_test, fmt = '%d')
        print('---split data done!---')
        print()
        clf = DecisionTreeClassifier(criterion = 'gini', random_state = 0) # 默认使用CART算法
        print(np.shape(X_train), np.shape(Y_train.ravel()))
        clf.fit(X_train, Y_train.ravel())
        # cross_val_score(classifier, X_train, Y_train, cv=5)
        # visualization
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
        graph.write_pdf("decision_tree.pdf")  
        # classifier.fit(X_train, Y_train)
        #验证测试集
        print("Detailed classification report:")  
        print()  
        print("The model is trained on the full development set.")  
        print("The scores are computed on the full evaluation set.")  
        print()
        target_names = ['lying', 'lie on the side', 'sitting', 'standing'] 
        Y_true, Y_pred = Y_test.ravel(), clf.predict(X_test)
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/Y_true.txt', Y_true, fmt = '%d')
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/Y_pred.txt', Y_pred, fmt = '%d')
        print(classification_report(Y_true, Y_pred, target_names = target_names))  
        print()

        print('Decision Tree model saving ......')
        model_save_path = '/home/hts/posture_classification_based_pose/decision_tree/train_decision_tree_model.m'
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
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/Y_true.txt', Y_true, fmt = '%d')
        np.savetxt('/home/hts/posture_classification_based_pose/decision_tree/Y_pred.txt', Y_pred, fmt = '%d')
        print(classification_report(Y_true, Y_pred, target_names = target_names))
        
        return
  
        

    def keyboard_control(self):

        command = self.command
        try:
            if command == 'predict_pic':
                self.predict_pic()
            elif command == 'training':
                self.decision_tree_training()
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
    parser.add_argument("--command", type = str, default = 'training',
                        help = "next command(default: 'training')")    
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
    train_decision_tree(keypoints_path, npsave_path, command, model_path)
