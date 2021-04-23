#!/usr/bin/env python
# coding: utf-8

# # Advertising dataset:classify users that would mostly click on an ad using Logistic regression & Decision tree classifier

# ## Table of contents

# * [Introduction](#Introduction)
# * [Data_wrangling](#Data_wrangling)
# * [Exploratory_Data_analysis](#Exploratory_Data_analysis)
# * [Model_building](#Model_building)
# * [Conclusions](#Conclusions)

# ## Introduction

# ## About Data

# In this project we will be working with an advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# In[2]:


#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[3]:


# Load data

df=pd.read_csv('advertising.csv')


# ## Data_wrangling

# In[4]:


df.info()


# Data consists of 1000 rows and 10 columns with data types of floats, ints,objects with no null values

# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


#checking for duplicates

df[df.duplicated()]


# In[8]:


#checking statistical insights

df.describe()


# checking for outiers in numerical columns

# In[9]:


df.head(1)


# In[10]:


# this is a function that takes a column name and calculates its IQR and min outlier and max outlier

def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range


# In[11]:


outlier_treatment(df['Daily Time Spent on Site'])


# In[12]:


# drawing a boxplot to check for outliers visually

sns.boxplot(df['Daily Time Spent on Site']);


# In[13]:


outlier_treatment(df['Area Income'])


# In[14]:


sns.boxplot(df['Area Income']);


# As per the boxplot it seems that we have -ve outliers that needs to be eliminated to give a better accuracy in our model

# In[15]:


outlier_treatment(df['Daily Internet Usage'])


# In[16]:


sns.boxplot(df['Daily Internet Usage']);


# In[17]:


outlier_treatment(df['Age'])


# In[18]:


sns.boxplot(df['Age']);


# In[19]:


#Removing -ve outliers from area income column

df=df[df['Area Income']>19300]


# In[20]:


#checking for final reults

df.info()


# ## Exploratory_Data_analysis

# In[21]:


df.describe()


# In[22]:


df.hist(['Daily Time Spent on Site','Age','Daily Internet Usage'],figsize=(16,24),layout=(5,2),alpha=0.75);


# Observations:
# * Age is concentrated more between the 30's
# * Daily time spend on site is mostly 80 minutes
# * Daiy Internet usage is usually mostly 225 minutes

# In[23]:


# pearson correlation calcuation

df[['Age','Area Income']].corr()


# In[24]:


# plotting a joint plot to show relation visually

sns.jointplot(x='Age',y='Area Income',data=df,kind='reg');


# As per the plot and the pearson correlation calculated of -0.18 , it seems that there is a -ve relation between Age and Area income as in the more the person gets old the less income he gets 

# In[25]:


# Creating a heatmap to find correlations amond data
plt.figure(figsize=(15,10));
sns.heatmap(df.corr(),annot=True,linewidths=0.5);


# Observations:
# 
# * Daily time spent on site and Daily internet used is moderally correlated(0.52)
# * Age and click on AD are moderally correlated (0.49)

# ## Model_building

# In[26]:


#checking columns i am going t include in my model

df.columns


# In[27]:


# Preparing test data

x=df[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage','Male']]
x.head(1)


# In[28]:


y=df['Clicked on Ad']


# In[29]:


#Splitting data into train and test data

from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test,y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)


# ##### Using Logistic Regression model

# In[31]:


#importing the model

from sklearn.linear_model import LogisticRegression


# In[32]:


logreg=LogisticRegression()


# In[33]:


#fitting the model

logreg.fit(X_train,y_train)


# In[34]:


#testing the model

predictions=logreg.predict(X_test)


# In[35]:


predictions


# In[36]:


y_test


# In[37]:


# Evaluating our model using calssification reprot

from sklearn.metrics import classification_report


# In[38]:


print(classification_report(y_test,predictions))


# In[39]:


from sklearn.metrics import confusion_matrix


# Classification report shows that the accuracy of the model is 92% which is a very good result, precision which represents the percentage of the positive predictive value is 0.89% in (0) will not click , 96% in (1) will click , recall
# which represents the percentage of relevant data found by the model in the dataset is 96% in (0) will not click , 89% in (1) will click

# In[40]:


confusion_matrix(y_test,predictions)


# ##### Using Descision Tree cassifier model

# In[41]:


from sklearn.tree import DecisionTreeClassifier


# In[42]:


dtree=DecisionTreeClassifier()


# In[43]:


dtree.fit(X_train,y_train)


# In[44]:


predictions_DT=dtree.predict(X_test)


# In[51]:


print(classification_report(y_test,predictions_DT))


# In[54]:


print(confusion_matrix(y_test,predictions_DT))


# ## Conclusions

# * Data frame has 1000 rows with 10 columns with no null values in any of the columns and its data types vary from float,int,objects
# * Data had no null values nor duplicated rows
# * There was some outliers in 'Area income' column which we removed them to enhance our model
# * Daily time spent on site and Daily internet used is moderally correlated(0.52)
# * Age and click on AD are moderally correlated (0.49)
# * As per the plot and the pearson correlation calculated of -0.18 , it seems that there is a -ve relation between Age and Area income as in the more the person gets old the less income he gets
# * Logistic Regression Classification report showed that the accuracy of the model is 92% which is a very good result
# * Logistic Regression Confusion matrix states that we got TP=140 Truely predicted , TN=135 Truely negative , FN=6 falsely negative and FP=17 falsely positive
# * Decision Tree Classifier Classification report showed that the f1 score of the model is 92% which is a very good result
# * Decision Tree Classifier Confusion matrix states that we got TP=139 Truely predicted , TN=143 Truely negative , FN=7 falsely negative and FP=9 falsely positive
