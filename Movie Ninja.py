
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[147]:


metadata=pd.read_csv("movie_metadata_with_score_metacritic.csv", index_col="Unnamed: 0") #reading in metacritic dataset
metadata=metadata.loc[~metadata["metacritic_metascore"].isna()] #removing rows with no metacritic data
metadata=metadata.drop(["homepage","keywords","overview","status","tagline","imdb_metascore"],1) #dropping unnecessary features
metadata=metadata.drop_duplicates() #dropping duplicates
metadata["release_year"]=metadata.release_date.str[0:4].astype(int) #creating new column with year for comparison


# In[131]:


rev_data=pd.read_csv("Revenue.csv") #reading in revenue data

#Adjusting units
rev_data["Budget"]=rev_data["Budget($M)"]*1000000
rev_data["Worldwide Gross"]=rev_data["Worldwide Gross($M)"]*1000000 
rev_data["Domestic Gross"]=rev_data["Domestic Gross($M)"]*1000000
rev_data=rev_data.drop(["Budget($M)","Domestic Gross($M)","Worldwide Gross($M)"],1)

rev_data=rev_data.rename(columns={"Movie":"title"})#renaming column to match metadata
rev_data=rev_data.drop_duplicates()#dropping duplicates


# In[102]:


critic_revenue=metadata.merge(rev_data,on="title") #merging datasets
critic_revenue=critic_revenue.drop_duplicates() #dropping duplicates
critic_revenue=critic_revenue.loc[(critic_revenue["Worldwide Gross"]!=0) | (critic_revenue["revenue"]!=0)] #removing rows with no revenue data
critic_revenue=critic_revenue.loc[(np.abs(critic_revenue.release_year-critic_revenue.Year)<=5)] #removing rows where the years don't match, as this indicates different movies

#Normalizing by the median for merged data set

critic_revenue["budget"]=(metadata["budget"]-metadata["budget"].median())/metadata["budget"].std()
critic_revenue["Worldwide Gross"]=(rev_data["Worldwide Gross"]-rev_data["Worldwide Gross"].median())/rev_data["Worldwide Gross"].std()

critic_revenue["Domestic Gross"]=(rev_data["Domestic Gross"]-rev_data["Domestic Gross"].median())/rev_data["Domestic Gross"].std()
critic_revenue["popularity"]=(metadata["popularity"]-metadata["popularity"].median())/metadata["popularity"].std()
critic_revenue["revenue"]=(metadata["revenue"]-metadata["revenue"].median())/metadata["revenue"].std()
critic_revenue["vote_count"]=(metadata["vote_count"]-metadata["vote_count"].median())/metadata["vote_count"].std()


