'''
Created on May 31, 2016

@author: pankajrawat
'''

from multiprocessing import freeze_support

from sklearn import cross_validation
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection.from_model import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from datetime import datetime
import os
import pandas as pd

pickle_dir = 'dataset'

def preprocess():
    print "Starting time => ", datetime.now()
    global df
    global dfLabel
    global dfTest
    global dfTestId
    global dfHoldOut
    global dfHoldOutLabel

    df = pd.read_csv(os.path.join(pickle_dir, 'data.csv'))
    dfLabel = df['defect']
    
    df.drop(['defect'], axis=1, inplace=True)

    dfHoldOut = None
    dfHoldOutLabel = None
    cv_pre = cross_validation.StratifiedShuffleSplit(dfLabel, 1, test_size=0.15, random_state=0)
    for train_index, test_index in cv_pre:
        print("TRAIN:", train_index, "TEST:", test_index)
        print train_index.max(), test_index.max()
        print df.shape, type(df)
        y_train, y_test = dfLabel[train_index], dfLabel[test_index]
        x_train, x_test = df.iloc[train_index], df.iloc[test_index]
        df, dfLabel = x_train, y_train
        dfHoldOut, dfHoldOutLabel = x_test, y_test
    print "==================== Data Set =================================="
    print "Holdout Set => ", dfHoldOut.shape
    print "Train Set => ", df.shape
    print "==================== Data Set =================================="


def train():
    tuned_params = {
                    
                     'kernel':  ['rbf', 'poly'],
                     'C': [0.01, 0.02, 0.05, 0.1, 1.0]
                }
    clf = SVC(probability=True)
    cv = cross_validation.StratifiedShuffleSplit(dfLabel, n_iter=3, test_size=0.2, random_state=1)
    gscv = GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=3, scoring="log_loss", n_jobs=3)

    print df.shape, dfLabel.shape
    gscv.fit(df, dfLabel)
    print gscv.best_estimator_, gscv.best_score_

    print "HoldOut score LLs => ", metrics.log_loss(dfHoldOutLabel, gscv.best_estimator_.predict_proba(dfHoldOut))
    print "HoldOut score Acc => ", metrics.accuracy_score(dfHoldOutLabel, gscv.best_estimator_.predict(dfHoldOut))
    
    classifiersDir = 'classifiers'
    estimator_pickefile = os.path.join(classifiersDir, gscv.best_estimator_.__class__.__name__)
    joblib.dump(gscv.best_estimator_, estimator_pickefile)


if __name__ == "__main__":
    freeze_support()
    preprocess()
    train()
#estimator_pickefile = os.path.join("classifiers", "SVC")
#estimator = joblib.load(estimator_pickefile)
#print estimator.support_vectors_, estimator.intercept_, estimator.coef_