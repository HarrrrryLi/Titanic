import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from collections import defaultdict
from collections import Counter

ages = defaultdict(int)

def extract_prefix(x):
  prefix = ''
  names = x.split(',')
  if len(names) > 1 and '.' in names[1]:
    prefix = names[1].split('.')[0]
    prefix = prefix.strip()
  return prefix

def simplify_name(df):
	df['Pre'] = df.Name.apply(lambda x: extract_prefix(x))
	return df

def statistics(df):
  counter = Counter()
  global ages
  for index,row in df.iterrows():
    if pd.notna(row['Age']):
      key = (row['Pre'], row['Sex'])
      counter[key] += 1
      ages[key] += row['Age']
  for k in ages:
    ages[k] /= counter[k]


def simplify_ages(df):
    statistics(df)
    global ages
    for index,row in df.iterrows():
      if pd.isna(row['Age']):
        key = (row['Pre'], row['Sex'])
        df.loc[index,'Age'] = ages[key]
    
    
    # bins = (0, 5, 12, 18, 25, 35, 60, 120)
    # group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young_Adult', 'Adult', 'Senior']
    # categories = pd.cut(df.Age, bins, labels=group_names)
    # df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_relationship(df):
	df['Relations'] = df.SibSp + df.Parch
	return df

def transform_features(df):
    df = simplify_name(df)
    df = simplify_relationship(df)
    df = simplify_ages(df)
    print(df)
    df = simplify_cabins(df)
    return df

def data_preprocess():
	dataframe=pd.read_csv('./train.csv')
	dataframe = transform_features(dataframe)
	targets = dataframe['Survived']
	feature_names = ['Pclass','Pre','Sex','Age','Relations','Cabin']
	features = dataframe[feature_names]

	for name in feature_names:
		features[name] = LabelEncoder().fit_transform(features[name])

	return features,targets
    



features,targets = data_preprocess()
num_test = 0.20
feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=num_test, random_state=23)
# print(features.shape)
# print(targets.shape)

clf = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)


grid_obj = grid_obj.fit(feature_train, target_train)


clf = grid_obj.best_estimator_
clf.fit(feature_train, target_train)

predictions = clf.predict(feature_test)
accuarcy = accuracy_score(target_test,predictions)
print(clf.feature_importances_)
print(accuarcy)
# print(target.shape)
# print(features.shape)