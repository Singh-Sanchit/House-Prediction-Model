
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits


# In[3]:


data = pd.read_csv("../Dataset/House_Data.csv")


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[7]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, height=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine


# In[8]:


plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")


# In[9]:


plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")


# In[10]:


plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")


# In[11]:


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


# In[12]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[13]:


plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")


# In[14]:


train1 = data.drop(['id', 'price'],axis=1)


# In[15]:


train1.head()


# In[16]:


data.floors.value_counts().plot(kind='bar')


# In[17]:


plt.scatter(data.floors,data.price)


# In[18]:


plt.scatter(data.condition,data.price)


# In[19]:


plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


reg = LinearRegression()


# In[22]:


labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)


# In[25]:


reg.fit(x_train,y_train)


# In[26]:


reg.score(x_test,y_test)


# In[27]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')


# In[28]:


clf.fit(x_train, y_train)


# In[29]:


clf.score(x_test,y_test)


# In[30]:


n_estimators = 400
t_sc = np.zeros(n_estimators,dtype=np.float64)


# In[31]:


y_pred = reg.predict(x_test)


# In[32]:


for i,y_pred in enumerate(clf.staged_predict(x_test)):
    t_sc[i]=clf.loss_(y_test,y_pred)


# In[33]:


testsc = np.arange(n_estimators)+1


# In[34]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')


# In[35]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[36]:


pca = PCA()


# In[43]:


pca.fit_transform(scale(train1))

