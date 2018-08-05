
# coding: utf-8

# In[1]:


get_ipython().system('pip install kaggle')


# In[ ]:


#from google.colab import files
#files.upload()


# In[ ]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[4]:


get_ipython().system('kaggle datasets download -d rishianand/devanagari-character-set')


# In[ ]:


import os

import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


root_dir = os.getcwd()
img_dir = os.path.join(root_dir, 'Images')


# In[ ]:


pixels = np.array(['pixel_{:04d}'.format(x) for x in range(1024)])
flag = True


# In[9]:


get_ipython().system('ls')


# In[ ]:


'''
for char_name in sorted(os.listdir(img_dir)):
    char_dir = os.path.join(img_dir, char_name)
    img_df = pd.DataFrame(columns=pixels)
    
    for img_file in sorted(os.listdir(char_dir)):
        image = pd.Series(imread(os.path.join(char_dir, img_file)).flatten(), index=pixels)
        img_df = img_df.append(image.T, ignore_index=True)
        
    img_df = img_df.astype(np.uint8)
    img_df['character'] = char_name
    
    img_df.to_csv('data.csv', index=False, mode='a', header=flag)
    flag=False
    
    print('=')
'''
    
    
df = pd.read_csv('data.csv')

df['character_class'] = LabelEncoder().fit_transform(df.character)
df.drop('character', axis=1, inplace=True)
df = df.astype(np.uint8)


# In[11]:


df.shape


# In[12]:


df_sample = df.sample(frac=0.1, random_state=0)

names = ['RidgeClassifier', 'BernoulliNB', 'GaussianNB', 'ExtraTreeClassifier', 'DecisionTreeClassifier',
         'NearestCentroid', 'KNeighborsClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier']
classifiers = [RidgeClassifier(), BernoulliNB(), GaussianNB(), ExtraTreeClassifier(), DecisionTreeClassifier(),
                NearestCentroid(), KNeighborsClassifier(), ExtraTreesClassifier(), RandomForestClassifier()]
test_scores, train_scores, fit_time, score_time = [], [], [], []

for clf in classifiers:
    scores = cross_validate(clf, df_sample.iloc[:, :-1], df_sample.iloc[:, -1])
    test_scores.append(scores['test_score'].mean())
    train_scores.append(scores['train_score'].mean())
    fit_time.append(scores['fit_time'].mean())
    score_time.append(scores['score_time'].mean())

pd.DataFrame({'Classifier': names,
              'Test_Score': test_scores,
              'Train_Score': train_scores,
              'Fit_Time': fit_time,
              'Score_Time': score_time})


# In[13]:


parameters = {'n_neighbors': np.arange(1, 22, 4)}
clf = GridSearchCV(KNeighborsClassifier(), parameters)

clf.fit(df_sample.iloc[:, :-1], df_sample.iloc[:, -1])
result = pd.DataFrame.from_dict(clf.cv_results_)

x, y = clf.best_params_['n_neighbors'], clf.best_score_
text = 'N Neighbors = {}, Score = {}'.format(x, y)

plt.figure()
plt.title('K Nearest Neighbors')
plt.xlabel('No. of Neighbors')
plt.ylabel('Accuracy Score')
plt.yticks(np.arange(0.6, 0.81, 0.02))

plt.plot(result.param_n_neighbors, result.mean_test_score, label='Mean Accuracy Score')
plt.plot(x, y, 'o', label=text)

plt.legend()
plt.show()


# In[14]:


parameters = {'n_estimators': np.arange(20, 310, 20)}
clf = GridSearchCV(ExtraTreesClassifier(), parameters)

clf.fit(df_sample.iloc[:, :-1], df_sample.iloc[:, -1])
result = pd.DataFrame.from_dict(clf.cv_results_)

x, y = clf.best_params_['n_estimators'], clf.best_score_
text = 'No. of Trees = {}, Score = {}'.format(x, y)

plt.figure()
plt.title('Extremely Randomized Trees Classification')
plt.xlabel('No. of Trees')
plt.ylabel('Accuracy Score')
plt.yticks(np.arange(0.6, 0.81, 0.02))

plt.plot(result.param_n_estimators, result.mean_test_score, label='Mean Accuracy Score')
plt.plot(x, y, 'o', label=text)

plt.legend()
plt.show()


# In[15]:


clf


# In[25]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('cp -i clf.pkl drive')

