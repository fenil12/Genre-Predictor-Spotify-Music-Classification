#!/usr/bin/env python
# coding: utf-8

# ## Its all about the Matrix! presented by 2B || !2B Team(CS985MLDAGroup15)
# #### Students Numbers :- 
# 202358886(Nishant Vimal)
# <br>
# 202370451(Fenil Patel)
# <br>
# 202394632(Rohit Satavekar)
# <br>
# 202358691(Chirdeep Singh Reen)
# <br>
# 202359355(Amit Pathak)
# <br>
# 202358578(Suditi Sharma)
# 
# <div style = "float: left; margin-right: 10px; padding-top: 5px">
#     <img src="https://media.tenor.com/c-I5YMwtnLoAAAAS/matrix-neo.gif" width="200" height="100">
# </div>

# #### Objective of this task:
# 
# The Aim of this problem is to predict the top genres using the classification of the spotify music data. The data is provided from the kaggle dataset:https://www.kaggle.com/competitions/cs9856-spotify-classification-problem-2023/data.
# <br>
# The challenge is to build a good machine learning model that is able to predict the genres.

# ## Importing Libraries
# 
# First, we import the libraries required to read csv file, to visualise the data and to calculate the performance of our model.

# In[1]:


#Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# Getting the train and test csv file and Loading it into Pandas Dataframe and below table shows the dataframe which we read from the csv file

# In[2]:


# Reading The CSV File and storing it in Pandas Dataframe
test1 = pd.read_csv('CS98XClassificationTest.csv')
train1 = pd.read_csv('CS98XClassificationTrain.csv')


# In[3]:


train1


# Identifying the datatype of our training data and finding the nulls in our dataset.

# In[4]:


train1.info()


# In[5]:


train1['top genre'].isnull().sum()


# To drop the null values, we use .dropna() method from pandas

# In[6]:


train1 = train1.dropna(axis=0)
train1


# # Analysing the data
# 
# <div style = "float: left; margin-right: 10px; padding-top: 5px">
#     <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjM3MjU2ZTRhNmIzMjVmN2EwNjA1YzU2MTNiNWM3NDk1Y2E0YTczNSZjdD1n/xT9C25UNTwfZuk85WP/giphy.gif" width="300" height="100">
# </div>
# 

# #### Plotting the pie-chart for the top 15 genres
# <br>
# The frequency of each genre in the training data set is shown here, along with the percentage of songs that fall into each genre among all the songs in the training data set that are included in the pie-chart.

# In[7]:


#Finding Frequency for the genres limited to 15
genres_categories = train1['top genre'].value_counts()[:15]
genres_categories


# In[8]:


#Calculating the percentage of genres
per_tg = genres_categories/len(train1)
per_tg


# In[9]:


#Taking keys and values per_tg
values = per_tg.values.tolist()
keys = per_tg.index.values.tolist()


# #### Plotting top genre Piechart

# In[10]:


#Plotting piechart
figure,axis = plt.subplots(figsize=(6,6))
axis.pie(values,labels=keys,autopct='%.2f%%',textprops={'fontsize':12}, startangle=25)
plt.title("Top 15 Popular Genres", bbox={'facecolor':'1.0', 'pad':5, 'alpha':1})


# #### Plotting the pie-chart for the top 15 artist
# Same as above , we are finding the percentage of songs belonging to respective artists out of all the songs present in training data set.

# In[11]:


#Finding Frequency for the artist limited to 15
artist_categories = train1['artist'].value_counts()[:15]
artist_categories


# In[12]:


#Calculating the percentage of artists
per_ar = artist_categories/len(train1)
per_ar


# In[13]:


#Taking keys and values per_ar
values = per_ar.values.tolist()
keys = per_ar.index.values.tolist()


# #### Plotting artist with most songs Piechart

# In[14]:


figure,axis = plt.subplots(figsize=(6,6))
axis.pie(values,labels=keys,autopct='%.2f%%',textprops={'fontsize':12}, startangle=25)
plt.title("Top 15 Artists with most songs", bbox={'facecolor':'1.0', 'pad':5, 'alpha':1})


# In[15]:


train1.query('artist == "Elton John"')


# In[16]:


train1.query('artist == "Queen"')


# In[17]:


train1.query('artist == "Taylor Swift"')


# ### Conclusion from our Visualisation and Reports
# Based on this we can associate a genre with an artist.
# <br>
# We think it would be useful to incorporate our artist column in our input features after analysing artists' musical genres.
# <br>
# On the basis of that, we'll use one hot encoding on artists.

# Plotting Linear Correlation Matrix

# In[18]:


import seaborn as sb

figure,axis = plt.subplots(figsize=(10,8))
sb.heatmap(train1.corr(), annot=True, cmap='magma')
plt.title('Correlation Matrix - Training Set')
plt.show


# #### What factors determine the genre of a song?
# <br>
# Based on the visualisation and reports above, we concluded that artists usually create songs of a specific genre. 
# Also, we have removed ID and title from the training set since it doesnt contribute to genre, but have included other features like tempo(bpm), words in song(spch), acousticness and other music variables that describe the genre of the songs. 

# # Implementing One hot encoding for the artist column
# 
# <div style = "float: left; margin-right: 10px; padding-top: 5px">
#     <img src="https://media.tenor.com/KBe_nw4IL2QAAAAC/matrix-code.gif" width="300" height="100">
# </div>

# Here we are implementing one hot encoding in "artist" column.
# <br>
# But the problem here is there are a lot more categories of artists in training set than test set while we are using pandas get_dummies method for one hot encoding.
# <br>
# To solve that problem, we first tried to get the mutual inclusive artist from the training and test set and then apply one hot encoding to both the training and test sets. The reason we took common categories is to maintain consistency while implementing one hot encoding.
# <br>
# After that, a left join is applied on the training data and the test data.
# <br>
# Then, we will align columns in the training and test dataframes in order to make both the dataframes identical.

# In[19]:


# Find the unique categories in the training and test data
train_categories = set(train1['artist'].unique())
test_categories = set(test1['artist'].unique())

# Find the common categories between the training and test data
common_categories = list(train_categories.intersection(test_categories))
common_categories


# In[20]:


# Replace the categorical feature in the training data with a numerical representation
# where each category is assigned a unique integer value
train1['artist'] = train1['artist'].replace(common_categories, range(len(common_categories)))

# Replace the categorical feature in the test data with a numerical representation
# where each category is assigned a unique integer value
test1['artist'] = test1['artist'].replace(common_categories, range(len(common_categories)))

# Perform one-hot encoding on the numerical representation of the categorical feature
train1 = pd.get_dummies(train1, columns=['artist'])
test1 = pd.get_dummies(test1, columns=['artist'])

# Ensure that the one-hot encoded data has the same number of columns in both the training and test data
train1, test1 = train1.align(test1, join='left', axis=1, fill_value=0)


# # Organizing training and test set
# 
# <div style = "float: left; margin-right: 10px; padding-top: 5px">
#     <img src="https://media.tenor.com/fK9_-Mxat30AAAAC/organize.gif" width="300" height="100">
# </div>

# In order to predict the target value("top genres") using the model that we have trained using training data set, the same features should be present in the test dataset as the training dataset.

# In[21]:


y_train = train1["top genre"]


# While doing one hot encoding, the test data also got top genres column from training set so we tried to drop that along with Id and title.

# In[22]:


#Dropping columns in training data
train1=train1.drop("Id",axis=1)
train1=train1.drop("top genre",axis=1)
train1=train1.drop("title",axis=1)

#Dropping columns in test data
test1=test1.drop("Id",axis=1)
test1=test1.drop("title",axis=1)
test1=test1.drop("top genre",axis=1)


# In[23]:


x_train = train1 #new_train1


# In[24]:


x_train


# In[25]:


x_test = test1
x_test


# # Training the Models
# 
# <div style = "float: left; margin-right: 10px; padding-top: 5px">
#     <img src="https://i.makeagif.com/media/7-05-2016/wf2hbR.gif" width="300" height="100">
# </div>

# To evaluate the model in our validation set, we divided the data as 80 percent in training data and 20 percent in validation.  
# <br>
# Here the random state = 42 is used, so that the same data gets repeated in training and validation set.
# <br>
# And using different values will results in different instances in training and validation set.

# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# To solve the classification problem, we have implemented several models including logistic regression, support vector machines, decision trees, random forests, and Extra Trees.
# <br>
# While evaluating, we know certain limitation of each models like logistic regression faces issues on non linear problems. SVM has issues on handling noisy data and being sensitive to imbalanced datasets. Decision tree issues related to bias and overfitting. Random forest and Extra Trees faces issue on sensitivity to noisy data
# <br>
# We tried separate models as it wasn't improving our score. So finally, We tried different approach using Voting Classifer since the different models were creating different errors and combining them, we expected better accuracy score. As a result, it improved our results. 

# In[27]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

log_clf = make_pipeline(StandardScaler(), LogisticRegression())
rnd_clf = RandomForestClassifier()
svc_clf = make_pipeline(StandardScaler(), SVC())
et_clf = ExtraTreesClassifier()
voting_clf = VotingClassifier(estimators=[('svc', svc_clf),('lr', log_clf), ('rf', rnd_clf), ('et', et_clf)],voting='hard')
voting_clf.fit(X_train, Y_train)


# In[28]:


y_pred_val = voting_clf.predict(X_val)


# # Evaluating the model
# 
# <div style = "float: left; margin-right: 10px; padding-top: 5px">
#     <img src="https://media.tenor.com/n4D40UqRwccAAAAC/morpheus-matrix.gif" width="300" height="100">
# </div>
# 

# We have used accuracy as our metric for evaluating the performance of our classification models.Out of the total number of samples, it calculates the percentage of correctly identified samples.
# <br>
# Other than accuracy, we have also used confusion matrix for evaluating the performance of our classification. Since it lists the total number of the model's accurate and inaccurate predictions, split down by each class.

# In[29]:


accuracy = accuracy_score(Y_val, y_pred_val)
print("Accuracy: ", accuracy)


# In[30]:


cm=confusion_matrix(Y_val,y_pred_val)
print("Confusion Matrix:\n ", cm)


# # Training the model in whole training set and Submitting our predicted results to kaggle

# In[31]:


voting_clf.fit(x_train, y_train)


# In[32]:


y_pred_test = voting_clf.predict(x_test)


# In[33]:


test_data_read = pd.read_csv('CS98XClassificationTest.csv')


# In[34]:


output_pred = pd.DataFrame({"id":test_data_read.Id,"top genre":y_pred_test})
output_pred.to_csv("submissionvoting2.csv",index=False)


# In[36]:


y_train


# # Performance on the Test set in Kaggle Competition

# It turns out based on our evaluation on validation set, it acted the same as in the test dataset in kaggle competition. Our score reaches upto 54% approx in the competition(https://www.kaggle.com/competitions/cs9856-spotify-classification-problem-2023/leaderboard). 
