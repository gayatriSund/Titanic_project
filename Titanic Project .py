#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv(r'\\SETHUSUND\Users\Sundar Ponni\Downloads\tested.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


#ADDING A COLUMN
data['aboard']=data['SibSp']+data['Parch']


# In[6]:


data.head()


# In[7]:


#FILLING MISSING VALUES
dat= data['Cabin'].fillna(0)


# In[8]:


data['Cabin']= dat


# In[9]:


data.head()


# In[10]:


data.isna().sum()


# In[11]:


#FILLING MISSING VALUES WITH MEAN
mean_age=data['Age'].mean()


# In[12]:


mean_age_ = data['Age'].fillna(mean_age)


# In[13]:


data['Age']=mean_age_


# In[14]:


data.tail()


# In[15]:


data.isna().sum()


# In[16]:


alt_fare = data['Fare'].fillna(0)
data['Fare']= alt_fare
data.isna().sum()


# In[17]:


# IDENTIFYING UNIQUE VALUES
data['Embarked'].unique()


# In[18]:


data['Sex'].unique()


# In[19]:


#MANAGING CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder
data['Sex']= LabelEncoder().fit_transform(data['Sex'])


# In[20]:


data['Embarked']=LabelEncoder().fit_transform(data['Embarked'])


# In[21]:


data.head()


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


# DROPPING COLUMNS
# STANDARD SCALING
data_dropped = data.drop(columns=['Name','Ticket','Cabin'])

data_scaled = pd.DataFrame(StandardScaler().fit_transform(data_dropped), columns=data_dropped.columns)
print(data_scaled)


# In[24]:


# NORMALISATION-IT IS TOO SENSITIVE TO OUTLIERS.(NORMALISATION/STANDARDISATION)
# IM RESETTING TO AVOID COMPLEXITY (TO AVOID THIS CELL)
from sklearn.preprocessing import MinMaxScaler
data_normalised= pd.DataFrame(MinMaxScaler().fit_transform(data_dropped), columns=data_dropped.columns)
print(data_normalised)


# In[25]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.hist(data['Age'])
plt.title('Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[26]:


import seaborn as sns
data=pd.DataFrame(data)
sns.pairplot(data)
plt.show()


# In[27]:


plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=10, kde=False)
plt.title('Histogram')
plt.show()


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Bar plot for survival counts
sns.countplot(x='Survived', data=data)
plt.title('Count of Survival on the Titanic')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


# In[29]:


plt.figure(figsize=(8, 6))
sns.histplot(x=data['Sex'])
plt.title('HISTPLOT')
plt.show()


# In[30]:


from sklearn.decomposition import PCA
# Initialize PCA
pca = PCA(n_components=2)

# Fit PCA to the preprocessed data
X_pca = pca.fit_transform(data_dropped)


# In[31]:


# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Survived'] = data['Survived'].reset_index(drop=True)


# In[32]:


# Plot PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2',data=pca_df, palette='Set1')
plt.title('PCA of Titanic Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Survived')
plt.show()


# In[33]:


# IT REDUCES THE FEATURES INTO TWO AND VISUALISES, PREVENTS VARIANCE & NOISE
print(pca_df)


# In[34]:


X= data_dropped.drop('Survived',axis=1)
y= data_dropped['Survived']


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[37]:


models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naive Bayes": GaussianNB()
}

# Dictionary to store accuracies
accuracies = {}

# Train, predict, and calculate accuracy for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

# Find and print the model with the highest accuracy
best_model = max(accuracies, key=accuracies.get)
print(f"\nBest Model: {best_model} with Accuracy: {accuracies[best_model]:.4f}")


# In[42]:


from sklearn.metrics import confusion_matrix
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[ ]:




