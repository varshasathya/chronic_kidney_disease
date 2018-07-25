
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
df=pd.read_csv("C:\\Users\\User\\Downloads\\kidney_disease.csv")
df.head()
#a = np.array(df)
#y  = a[0:25]
#print(y)


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm


# In[3]:


df.keys()


# In[4]:


df=df.replace(' ',np.nan)
df=df.replace('\t?',np.nan)
df.head()


# In[5]:


df = df.dropna(axis=0, how="any")


# In[6]:


df.info()


# In[7]:


df[['pcv','wc','rc']] = df[['pcv','wc','rc']].apply(pd.to_numeric)
df.info()


# In[8]:


df = pd.get_dummies(df)
df.info()


# In[43]:


from sklearn.model_selection import train_test_split
"""X_train = df.drop("classification", axis=1)
Y_train = df["classification"]
X_test  = df.drop("age", axis=1).copy()"""
X_train, X_test, y_train, y_test = train_test_split(df, df['classification'], test_size=0.30, random_state=101)



# In[42]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
"""Y_pred = model.predict(X_test)
acc_svc = round(model.score(X_train, Y_train) * 100, 2)
acc_svc
Y_pred = model.predict(X_test)\nacc_svc = round(model.score(X_train, Y_train) * 100, 2)\nacc_svc"""


# In[25]:


predictions = model.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report,confusion_matrix


# In[27]:


print(confusion_matrix(y_test,predictions))


# In[28]:


print(classification_report(y_test,predictions))


# In[29]:


from sklearn.model_selection import GridSearchCV


# In[30]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[31]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# In[32]:


grid.best_params_


# In[33]:


grid.best_estimator_


# In[34]:


grid_predictions = grid.predict(X_test)


# In[35]:


print(confusion_matrix(y_test,grid_predictions))


# In[36]:


print(classification_report(y_test,grid_predictions))

