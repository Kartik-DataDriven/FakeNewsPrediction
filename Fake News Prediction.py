#!/usr/bin/env python
# coding: utf-8

# ### Exploring the World of ML: Fake News Prediction Project by Kartik (Age 13) 🚀📰
# 
# In this Jupyter Notebook, join me on a journey into the fascinating realm of machine learning as I delve into a project focused on predicting fake news. Together, we'll unravel the intricacies of algorithms, analyze news articles, and strive to distinguish fact from fiction. Stay tuned for updates and insights into my exploration of AI and misinformation detection! 👨‍💻✨

# #### Importing Important libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import warnings 

warnings.simplefilter("ignore")


# ### Importing Dataset

# In[2]:


df_real = pd.read_csv('./dataset/True.csv')
df_fake = pd.read_csv('./dataset/Fake.csv')


# In[3]:


df_real.head()


# In[4]:


df_fake.head()


# In[5]:


df_fake.shape, df_real.shape


# In[6]:


# adding a column called class
df_real["class"] = 1
df_fake["class"] = 0


# ### Preparing portion of dataset for Manual Testing
# 

# In[7]:


# Creating manual testing datasets
df_fake_manual_testing = df_fake.tail(10)
df_real_manual_testing = df_real.tail(10)

# Removing last 10 rows from training datasets
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)
    
for i in range(21416, 21406, -1):
    df_real.drop([i], axis=0, inplace=True)


# In[8]:


df_fake_manual_testing.head()


# In[9]:


df_real_manual_testing.head()


# In[10]:


df_fake_manual_testing["class"] = 0
df_real_manual_testing["class"] = 1


# #### Exporting the dataset which will be convenient in the future for us

# In[11]:


# Combining manual testing datasets
manual_testing_data = pd.concat([df_fake_manual_testing, df_real_manual_testing], axis=0)

# Saving manual testing data to a CSV file
manual_testing_data.to_csv("./dataset/manual_testing.csv")


# ## Merging df_real and df_fake into one single dataset

# In[12]:


df = pd.concat([df_fake, df_real], axis=0)
df.head()


# In[13]:


df.columns


# #### Checking Null values (luckily we don't have any null values)

# In[14]:


df.isnull().sum()


# In[15]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

df.columns


# In[16]:


df.head()


# In[17]:


df


# ## Importing Important Libraries for the model creation

# In[18]:


import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ## Creating Function for Text Preprocessing 

# In[19]:


def textpreprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# ## Applying text preprocessing to 'title,' 'text,' and 'subject'

# In[20]:


df['processed_title'] = df['title'].apply(textpreprocess)
df['processed_text'] = df['text'].apply(textpreprocess)
df['processed_subject'] = df['subject'].apply(textpreprocess)

# Combine processed text columns into a single column
df['combined_processed_text'] = (
    df['processed_title'] + ' ' +
    df['processed_text'] + ' ' +
    df['processed_subject']
)


# ## Vectorize Text Data

# In[21]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
text_matrix = tfidf_vectorizer.fit_transform(df['combined_processed_text'])


# ## Split into Features and Target

# In[22]:


X = text_matrix
y = df['class']

# Step 4: Split into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Train Logistic Regression Model

# In[23]:


logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Make Predictions and Evaluate

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# classification_report_result = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:\n", classification_report_result)


# ## Train Random Forest Model

# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print(f"Random Forest Accuracy: {accuracy:.2f}")


# ## Manual Testing

# In[25]:


def output_label(n):
    if n == 0:
        return "It's a Fake News"
    elif n == 1:
        return "It's a Real News"
    
def manual_testing(news, logreg_model, random_forest_model, tfidf_vectorizer):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    # Assuming no separate text preprocessing is needed
    new_x_test = new_def_test["text"]
    # Assuming no separate vectorization is needed
    new_xv_test = tfidf_vectorizer.transform(new_x_test)  # Make sure to replace 'tfidf_vectorizer' with your actual vectorization method

    y_pred_logreg = logreg_model.predict(new_xv_test)
    y_pred_rf = random_forest_model.predict(new_xv_test)

    print("\n\nLogistic Regression Prediction: {}".format(output_label(y_pred_logreg[0])))
    print("Random Forest Prediction: {}".format(output_label(y_pred_rf[0])))

# Example usage:
news = str(input("Enter news: "))
manual_testing(news, logreg_model, random_forest_model, tfidf_vectorizer)


# ## Exporting the model for Creating USER INTERFACE

# In[30]:


import pickle

model_save_path = "./model/logisticreg_model.pkl"
# Save the model using pickle
with open(model_save_path, 'wb') as file:
    pickle.dump(logreg_model, file)


# In[ ]:




