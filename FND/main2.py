import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, recall_score ,precision_score
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
#from sklearn.metrics import average_precision_score

def plot_confusion(conf,s):
    labels=['Fake','True']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf)
    plt.title('Confusion matrix of the classifier\n')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('PREDICTED')
    plt.ylabel('ACTUAL')
    plt.savefig(os.getcwd() + "/media/" +s)
    plt.show()
	

def build_confusion_matrix(classifier,df,s,linegraph):
    k_fold = KFold(n_splits=10)
    y_real = []
    y_proba = []
    i=0
    scores = []
    recalls =[]
    precisions = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(df):        
        train_text =df.iloc[train_ind]['Statement'] 
        train_y = df.iloc[train_ind]['Label']    
        test_text = df.iloc[test_ind]['Statement']
        test_y = df.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        scores.append(f1_score(test_y, predictions))
        recalls.append(recall_score(test_y, predictions))
        precisions.append(precision_score(test_y, predictions))
        
        i=i+1
        precisioncc, recallcc, _ = precision_recall_curve(test_y, predictions)
        lab = 'Fold %d AUC=%.4f' % (i, auc(recallcc, precisioncc))
        
        
        y_real.append(test_y)
        y_proba.append(predictions)
        
        #plt.subplot(11, 1, i)
        plt.step(recallcc, precisioncc, label=lab)

    score = sum(scores) / len(scores)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    
    precisioncc, recallcc, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUC=%.4f' % (auc(recallcc, precisioncc))
    
    #plt.subplot(11, 1, i)
    plt.step(recallcc, precisioncc, label=lab, lw=2, color='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize='small')
    plt.savefig(os.getcwd() + "/media/" +linegraph)
    
    plot_confusion(confusion,s)

    return (len(df),confusion,score,precision,recall)
	
	
countV = CountVectorizer()
tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)


def callClass(classifier1,dataf,title,s):
        print('\n'+title+'\n')
        conMat = np.array([[0,0],[0,0]])
        NoStat, conMat, f1score, prec, rec = build_confusion_matrix(classifier1,dataf,s,'curve_'+s)
        		 
        print('Total statements classified:', NoStat)
        print('Confusion matrix:')
        print(conMat)
        print('F1 Score:', f1score)
        print('Precision:',prec)
        print('Recall:',rec)

        with open(os.path.dirname(__file__)+'/results.csv','a') as file:
            file.write(str(NoStat)+','+str(conMat)+','+str(f1score)+','+str(prec)+','+str(rec))
            file.write("\n")
            file.close()

'''    
#plotting Precision-Recall curve
def plot_PR_curve(classifier,DataPrep,title):
    X = DataPrep['Statement']
    y = DataPrep['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    
    y_train= y_train.astype(int)
    y_test = y_test.astype(int)
    precision, recall, thresholds = precision_recall_curve(y_test, classifier)
    average_precision = average_precision_score(y_test,classifier)
    
    plt.step(recall, precision, color='r', alpha=0.2,
             where='post')
    #plt.fill_between(recall, precision, step='post', alpha=0.2,color='g')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title +' Precision-Recall curve: AP={0:0.2f}'.format(average_precision))'''
   
       
def LogReg(data):
    
    '''X = data['Statement']
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
    y_train= y_train.astype(int)
    y_test = y_test.astype(int)'''
    
    logR_pipeline = Pipeline([
    						('LogRCV',countV),
    						('LogR_clf',LogisticRegression(solver='liblinear'))
    						])
    
  
    
    
    callClass(logR_pipeline,data,'LOGISTIC REGRESSION - USING TF FEATURES','LOGISTIC REGRESSION.png')
  
    ''' logR_pipeline.fit(X_train,y_train)
    predicted_LogR = logR_pipeline.predict(X_test)
    np.mean(predicted_LogR == y_test)
    plot_PR_curve(predicted_LogR,data,'LOGISTIC REGRESSION')
    '''
    logR_pipeline_ngram = Pipeline([
                        ('LogR_tfidf',tfidf_ngram),
                        ('LogR_clf',LogisticRegression(solver='liblinear',penalty="l2",C=1))
                        ])
  
    callClass(logR_pipeline_ngram,data,'LOGISTIC REGRESSION - USING TF-IDF FEATURES','LOGISTIC REGRESSION - USING TF-IDF FEATURES.png')
    #plot_learing_curve(logR_pipeline_ngram,"LogisticRegression Classifier",data) 
       
    '''logR_pipeline_ngram.fit(X_train,y_train)
    predicted_LogR_ngram = logR_pipeline_ngram.predict(X_test)
    np.mean(predicted_LogR_ngram == y_test)
    plot_PR_curve(predicted_LogR_ngram,data,'LOGISTIC REGRESSION')'''
    

def NaiveBay(data):

        ''' X = data['Statement']
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
        y_train= y_train.astype(int)
        y_test = y_test.astype(int)'''
    
        nb_pipeline = Pipeline([
        						('NBCV',countV),
        						('nb_clf',MultinomialNB())])
        	
        callClass(nb_pipeline,data,'NAIVE BAYES - USING TF FEATURES','NAIVE BAYES - USING TF FEATURES.png')
        
        ''' nb_pipeline.fit(X_train,y_train)
        predicted_nb = nb_pipeline.predict(X_test)
        np.mean(predicted_nb == y_test)
        plot_PR_curve(predicted_nb,data,'NAIVE BAYES')'''
        
        nb_pipeline_ngram = Pipeline([
                            ('nb_tfidf',tfidf_ngram),
                            ('nb_clf',MultinomialNB())])
       
        callClass(nb_pipeline_ngram,data,'NAIVE BAYES - USING TF-IDF FEATURES','NAIVE BAYES - USING TF-IDF FEATURES.png')
        
        ''' nb_pipeline_ngram.fit(X_train,y_train)
        predicted_nbgram = nb_pipeline_ngram.predict(X_test)
        np.mean(predicted_nbgram == y_test)
        plot_PR_curve(predicted_nbgram,data,'NAIVE BAYES')
        '''

def svmClass(data):
    
        '''X = data['Statement']
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
        y_train= y_train.astype(int)
        y_test = y_test.astype(int)'''
        
        svm_pipeline = Pipeline([
                            ('svmCV',countV),
                            ('svm_clf',svm.LinearSVC())
                            ])
              
        callClass(svm_pipeline,data,'SVM - USING TF FEATURES','SVM - USING TF FEATURES.png')
        
        '''svm_pipeline.fit(X_train,y_train)
        predicted_svm = svm_pipeline.predict(X_test)
        np.mean(predicted_svm == y_test)
        plot_PR_curve(predicted_svm,data,'SVM')'''
                        
        svm_pipeline_ngram = Pipeline([
                            ('svm_tfidf',tfidf_ngram),
                            ('svm_clf',svm.LinearSVC())
                            ])
        
        callClass(svm_pipeline_ngram,data,'SVM - USING TF-IDF FEATURES','SVM - USING TF-IDF FEATURES.png')
        
        '''svm_pipeline_ngram.fit(X_train,y_train)
        predicted_svmgram = svm_pipeline_ngram.predict(X_test)
        np.mean(predicted_svmgram == y_test)
        plot_PR_curve(predicted_svmgram,data,'SVM')'''


def decisionTree(data):
        '''X = data['Statement']
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        y_train= y_train.astype(int)
        y_test = y_test.astype(int)'''

        decision_pipeline = Pipeline([
            ('decisionCV', countV),
            ('Decision_clf', DecisionTreeClassifier())
        ])

        callClass( decision_pipeline, data, 'decision- USING TF FEATURES', 'decision - USING TF FEATURES.png')

        '''svm_pipeline.fit(X_train,y_train)
        predicted_decision = decision_pipeline.predict(X_test)
        np.mean(predicted_decision == y_test)
        plot_PR_curve(predicted_decision,data,'decision')'''

        decision_pipeline_ngram = Pipeline([
            ('decision_tfidf', tfidf_ngram),
            ('decision_clf', DecisionTreeClassifier())
        ])

        callClass( decision_pipeline_ngram, data, 'decision - USING TF-IDF FEATURES', 'decision- USING TF-IDF FEATURES.png')

        '''decision_pipeline_ngram.fit(X_train,y_train)
        predicted_decisiongram = decision_pipeline_ngram.predict(X_test)
        np.mean(predicted_decisiongram == y_test)
        plot_PR_curve(predicted_decisiongram,data,'decision')'''


def knnClass(data):
        '''X = data['Statement']
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        y_train= y_train.astype(int)
        y_test = y_test.astype(int)'''

        knn_pipeline = Pipeline([
            ('knnCV', countV),
            ('knn_clf', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))
        ])

        callClass(knn_pipeline, data, 'knn - USING TF FEATURES', 'knn - USING TF FEATURES.png')



        knn_pipeline_ngram = Pipeline([
            ('knn_tfidf', tfidf_ngram),
            ('knn_clf',  KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))
        ])

        callClass(knn_pipeline_ngram, data, 'knn - USING TF-IDF FEATURES', 'knn - USING TF-IDF FEATURES.png')



def first(df):
   LogReg(df)
   NaiveBay(df)
   svmClass(df)
   decisionTree(df)
   knnClass(df)
   

