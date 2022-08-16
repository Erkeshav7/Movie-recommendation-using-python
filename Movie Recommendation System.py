#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import warnings


# In[34]:


warnings.filterwarnings('ignore')


# In[35]:


columns_names=["user_id","item_id","rating","timestamp"]
df=pd.read_csv("u.data",sep='\t',names=columns_names)


# In[21]:


df.head()


# In[36]:


df.shape


# In[37]:


df["user_id"].nunique()


# In[38]:


df["item_id"].nunique()


# In[39]:


movies_titles=pd.read_csv("u.item",sep='\|',header=None)


# In[40]:


movies_titles=movies_titles[[0,1]]


# In[43]:


movies_titles.columns=["item_id","title"]
movies_titles.head()


# In[45]:


df=pd.merge(df,movies_titles,on="item_id")


# In[46]:


df.tail()


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# In[50]:


df.groupby('title').mean()['rating'].sort_values(ascending=False)


# In[52]:


df.groupby("title").count()['rating'].sort_values(ascending=False)


# In[56]:


ratings=pd.DataFrame(df.groupby("title").mean()['rating'])


# In[57]:


ratings.head()


# In[60]:


ratings['num of ratings'] = pd.DataFrame(df.groupby("title").count()['rating'])


# In[61]:


ratings


# In[63]:


plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'],bins = 70)
plt.show()


# In[65]:


plt.hist(ratings['rating'],bins=70)
plt.show()


# In[71]:


sns.jointplot(x='rating' ,y='num of ratings' ,data=ratings,alpha=0.5)


# In[72]:


df.head()


# In[74]:


movie_matrix=df.pivot_table(index="user_id",columns="title",values="rating")


# In[77]:


movie_matrix.head()


# In[79]:


ratings.sort_values('num of ratings',ascending=False).head()


# In[82]:


starwars_user_rating=movie_matrix['Star Wars (1977)']
starwars_user_rating.head()


# In[87]:


similar_to_starwars=movie_matrix.corrwith(starwars_user_rating)
corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlations'])


# In[89]:


corr_starwars.dropna(inplace= True)


# In[90]:


corr_starwars


# In[93]:


corr_starwars.sort_values('correlations',ascending=False).head(10)


# In[97]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[98]:


corr_starwars.head()


# In[100]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlations',ascending=False)


# In[109]:


def predict_movies(movie_name):
    movie_user_ratings=movie_matrix[movie_name]
    similar_to_movie=movie_matrix.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlations'])
    corr_movie.dropna(inplace=True)
    corr_movie= corr_movie.join(ratings['num of ratings'])
    predictions= corr_movie[corr_movie['num of ratings']>100].sort_values('correlations',ascending=False)
    return predictions
    


# In[110]:


predictions=predict_movies("Titanic (1997)")


# In[111]:


predictions.head()

