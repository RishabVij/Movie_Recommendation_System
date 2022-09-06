#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')


# In[4]:


movies.head(2)


# In[5]:


credits.head(2)


# In[6]:


credits.head(1)['cast'].values


# In[7]:


movies = movies.merge(credits,on = 'title')


# In[8]:


movies.head()


# In[9]:


credits.shape


# In[10]:


movies['original_language'].value_counts()


# In[11]:


movies.info()


# In[12]:


#genres
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[13]:


movies.head(3)


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace = True)


# In[16]:


movies.isnull().sum()


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres


# In[19]:


import ast


# In[20]:


def convert(obj):   
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[21]:


movies['genres'] = movies['genres'].apply(convert)


# In[22]:


movies.head(2)


# In[23]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[24]:


movies.head(2)


# In[25]:


movies['cast'][0]


# In[26]:


def convert3(obj):   
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[27]:


movies['cast'].apply(convert3)


# In[28]:


movies['cast'] = movies['cast'].apply(convert3)


# In[29]:


movies.head(2)


# In[30]:


movies['crew'][0]


# In[31]:


def fetchdirector(obj):   
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[32]:


movies['crew'] = movies['crew'].apply(fetchdirector)


# In[33]:


movies.head(2)


# In[34]:


movies['overview'][0]


# In[35]:


movies['overview'].apply(lambda x:x.split())


# In[36]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[37]:


movies.head(2)


# In[38]:


movies['genres'].apply(lambda x:[i.replace(" ","") for i in x] )


# In[39]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x] )


# In[40]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x] )


# In[41]:


movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x] )


# In[42]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x] )


# In[43]:


movies.head(3)


# In[44]:


movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[45]:


movies.head()


# In[46]:


new_df = movies[['movie_id','title','tags']]


# In[47]:


new_df.head()


# In[48]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[49]:


new_df.head()


# In[50]:


new_df['tags'][0]


# In[51]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[52]:


new_df.head()


# In[53]:


from sklearn.feature_extraction.text import CountVectorizer 


# In[54]:


cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[55]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[56]:


vectors


# In[57]:


vectors[0]


# In[58]:


cv.get_feature_names()


# In[59]:


import nltk


# In[60]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[61]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[62]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[63]:


cv.get_feature_names()


# In[64]:


from sklearn.metrics.pairwise import cosine_similarity


# In[65]:


similarity = cosine_similarity(vectors)


# In[66]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True, key = lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    


# In[67]:


movies.head()


# In[68]:


recommend('The Dark Knight Rises')


# In[69]:


import pickle


# In[73]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[74]:


pickle.dump(similarity,open('similarity','wb'))


# In[ ]:





# In[ ]:





# In[ ]:




