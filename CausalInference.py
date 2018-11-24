import numpy as np
import pandas as pd
import json

# Get the top 4 actors/cast of the movie and the writer and director of the movie

credits=pd.read_csv("tmdb_5000_credits.csv")
for movie_num in range(len(credits)):
    print(movie_num)
    cast_json = json.loads(credits.loc[movie_num,'cast'])
    crew_json = json.loads(credits.loc[movie_num,'crew'])
    for index_json in range(min(4,len(cast_json))):
        credits.loc[movie_num,cast_json[index_json]['name']] = 1
    for crew in crew_json:
        if crew['job'].lower().strip()== 'director':
            credits.loc[movie_num,crew['name']] = 1
        elif crew['job'].lower().strip()== 'writer':
            credits.loc[movie_num,crew['name']] = 1

for column in credits.columns:
    credits[column].fillna(0,inplace=True)

credits.drop(columns=['cast','crew','title'])
credits.to_csv("unpacked_tmdb_credits.csv")

# Unchanged code from before - removed the normailizing part as it was not required for any methods used.
# Also all the production houses have been kept as it is since 
# creating a big production house variable is really difficult as the production house names 
# don't match for every movie as they can be subsidiaries or just abbreviated names and all

metadata=pd.read_csv("movie_metadata_with_score_metacritic.csv", index_col="Unnamed: 0") #reading in metacritic dataset
metadata=metadata.loc[~metadata["metacritic_metascore"].isna()] #removing rows with no metacritic data
metadata=metadata.drop(["homepage","keywords","overview","status","tagline","imdb_metascore","budget","revenue"],1) #dropping unnecessary features
metadata=metadata.drop_duplicates() #dropping duplicates
metadata["release_year"]=metadata.release_date.str[0:4].astype(int) #creating new column with year for comparison

num_data=pd.read_csv("the_numbers_budget.csv") #reading in revenue data
num_data=num_data.rename(columns={"Movie":"title"}) #renaming column to match metadata
num_data["num_year"]=num_data["Release Date"].str[-4:].astype(int)
num_data=num_data.drop_duplicates()#dropping duplicates

critic_revenue=metadata.merge(num_data,on="title") #merging datasets
critic_revenue=critic_revenue.drop_duplicates() #dropping duplicates
critic_revenue=critic_revenue.loc[(critic_revenue["Worldwide Gross"]!=0)] #removing rows with no revenue data
critic_revenue=critic_revenue.loc[(np.abs(critic_revenue.release_year-critic_revenue.num_year)<5)] #removing rows where the years don't match, as this indicates different movies

# Reindex since the dataset has been subset many times
critic_revenue.index=range(len(critic_revenue))

#These are the columns we may not model on
keep_list=critic_revenue.columns

# Goal is to create dummy variables from different columns with json string

for movie_num in range(len(critic_revenue)):
    flat_json=json.loads(critic_revenue.loc[movie_num,'production_companies'])
    for index_json in range(len(flat_json)):
        critic_revenue.loc[movie_num,flat_json[index_json]['name']] = 1

prod_company_list = [column for column in critic_revenue.columns if column not in keep_list]        

for movie_num in range(len(critic_revenue)):
    flat_json=json.loads(critic_revenue.loc[movie_num,'genres'])
    for index_json in range(len(flat_json)):
        critic_revenue.loc[movie_num,flat_json[index_json]['name']] = 1

genres_list = [column for column in critic_revenue.columns if column not in keep_list and column not in prod_company_list]        

critic_cast_revenue=critic_revenue.merge(credits['cast','crew'],left_on ="id",right_on="movie_id",how="left")

model_cols = [column for column in critic_cast_revenue.columns if column not in keep_list]

# Need to fill the NaN/Missing values with 0 for dummy variable
for column in model_cols:
    critic_cast_revenue[column].fillna(0,inplace=True)

# The positive, mixed and negative review tags are based on the metascore description given by metacritic
critic_cast_revenue['Good Critic Review'] = [1 if critic_cast_revenue.loc[i,'metacritic_metascore'] > 60 else 0 if critic_cast_revenue.loc[i,'metacritic_metascore'] > 40 else -1 for i in range(len(critic_cast_revenue)) ]

model_cols.append('Production Budget')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor as knn_reg

log_model = LogisticRegression()

# Randomly divide the movies with positive and mixed review data into 2 parts, half for training and half for prediction.

good_review = critic_cast_revenue.loc[critic_cast_revenue['Good Critic Review']>=0,:].copy()
good_review.loc[:,'random'] = np.random.random(len(good_review))
train_log = good_review.loc[good_review['random']<=0.5,:].copy()
test_log = good_review.loc[good_review['random']>0.5,:].copy()

# Model Propensity score i.e., probability to get a good review
# using Logistic regression on the positive and mixed reviews training data.

log_model.fit(train_log.loc[:,model_cols],train_log['Good Critic Review'])

# Predict the propensity score on the test data

test_log.loc[:,'prop_score'] = log_model.predict_proba(test_log.loc[:,model_cols])[:,1]

# Matching Using 1 Nearest Neighbor based on only Propensity Score for the test data i.,e
# match every movie with good review to a movie with mixed review to get an estimate of it's revenue
# had it received mixed reviews and vice-versa for movies with mixed reviews

good_review_test = test_log.loc[test_log['Good Critic Review']==1,:].copy()
neutral_review_test = test_log.loc[test_log['Good Critic Review']==0,:].copy()
knn_model = knn_reg(n_neighbors=1)
knn_model.fit(np.array(good_review_test['prop_score']).reshape(-1,1),good_review_test.loc[:,"Worldwide Gross"])
neutral_review_test.loc[:,'good_revenue'] = knn_model.predict(np.array(neutral_review_test.loc[:,'prop_score']).reshape(-1,1))
knn_model.fit(np.array(neutral_review_test.loc[:,'prop_score']).reshape(-1,1),neutral_review_test.loc[:,"Worldwide Gross"])
good_review_test.loc[:,'neutral_revenue'] = knn_model.predict(np.array(good_review_test.loc[:,'prop_score']).reshape(-1,1))

# Paired t-test to get significance of the difference in revenue due to a positive review vs a mixed review

t_stat_good_neutral,p_good_neutral = stats.ttest_rel(good_review_test['neutral_revenue'],good_review_test['Worldwide Gross'])
print(p_good_neutral)
t_stat_neutral_good,p_neutral_good = stats.ttest_rel(neutral_review_test['good_revenue'],neutral_review_test['Worldwide Gross'])
print(p_neutral_good)

# Randomly divide the movies with negative and mixed review data into 2 parts, half for training and half for prediction.

bad_review = critic_cast_revenue.loc[critic_cast_revenue['Good Critic Review']<=0,:].copy()
bad_review['Bad Critic Review'] = bad_review['Good Critic Review'].replace(-1,1) 
bad_review.loc[:,'random'] = np.random.random(len(bad_review))
train_log = bad_review.loc[bad_review['random']<=0.5,:].copy()
test_log = bad_review.loc[bad_review['random']>0.5,:].copy()

# Model Propensity score i.e., probability to get a negative review
# using Logistic regression on the negative and mixed reviews training data.

log_model.fit(train_log.loc[:,model_cols],train_log['Bad Critic Review'])
test_log.loc[:,'prop_score'] = log_model.predict_proba(test_log.loc[:,model_cols])[:,1]

# Matching Using 1 Nearest Neighbor based on only Propensity Score for the test data i.,e
# match every movie with negative review to a movie with mixed review to get an estimate of it's revenue
# had it received mixed reviews and vice-versa for movies with mixed reviews

neutral_review_test = test_log.loc[test_log['Bad Critic Review']==0,:].copy()
bad_review_test = test_log.loc[test_log['Bad Critic Review']==1,:].copy()
knn_model = knn_reg(n_neighbors=1)
knn_model.fit(np.array(bad_review_test['prop_score']).reshape(-1,1),bad_review_test.loc[:,"Worldwide Gross"])
neutral_review_test.loc[:,'bad_revenue'] = knn_model.predict(np.array(neutral_review_test.loc[:,'prop_score']).reshape(-1,1))
knn_model.fit(np.array(neutral_review_test.loc[:,'prop_score']).reshape(-1,1),neutral_review_test.loc[:,"Worldwide Gross"])
bad_review_test.loc[:,'neutral_revenue'] = knn_model.predict(np.array(bad_review_test.loc[:,'prop_score']).reshape(-1,1))

# Paired t-test to get significance of the difference in revenue due to a negative review vs a mixed review

t_stat_bad_neutral,p_bad_neutral = stats.ttest_rel(bad_review_test['neutral_revenue'],bad_review_test['Worldwide Gross'])
print(p_bad_neutral)
t_stat_neutral_bad,p_neutral_bad = stats.ttest_rel(neutral_review_test['bad_revenue'],neutral_review_test['Worldwide Gross'])
print(p_neutral_bad)

# Model Propensity score i.e., probability to get a positive review
# using Logistic regression on negative and positive reviews training data.

bad_review = critic_cast_revenue.loc[(critic_cast_revenue['Good Critic Review']==-1) | (critic_cast_revenue['Good Critic Review']==1),:].copy()
bad_review.loc[:,'Good Critic Review'].replace(-1,0,inplace=True) 
bad_review.loc[:,'random'] = np.random.random(len(bad_review))
train_log = bad_review.loc[bad_review['random']<=0.5,:].copy()
test_log = bad_review.loc[bad_review['random']>0.5,:].copy()
log_model.fit(train_log.loc[:,model_cols],train_log['Good Critic Review'])
test_log.loc[:,'prop_score'] = log_model.predict_proba(test_log.loc[:,model_cols])[:,1]

# Matching Using 1 Nearest Neighbor based on only Propensity Score for the test data i.e.,
# match every movie with negative review to a movie with positive review to get an estimate of it's revenue
# had it received positive reviews and vice-versa for movies with positive reviews

good_review_test = test_log.loc[test_log['Good Critic Review']==1,:].copy()
bad_review_test = test_log.loc[test_log['Good Critic Review']==0,:].copy()
knn_model = knn_reg(n_neighbors=1)
knn_model.fit(np.array(bad_review_test['prop_score']).reshape(-1,1),bad_review_test.loc[:,"Worldwide Gross"])
good_review_test.loc[:,'bad_revenue'] = knn_model.predict(np.array(good_review_test.loc[:,'prop_score']).reshape(-1,1))
knn_model.fit(np.array(good_review_test.loc[:,'prop_score']).reshape(-1,1),good_review_test.loc[:,"Worldwide Gross"])
bad_review_test.loc[:,'good_revenue'] = knn_model.predict(np.array(bad_review_test.loc[:,'prop_score']).reshape(-1,1))

# Paired t-test to get significance of the difference in revenue due to a negative review vs a positive review

t_stat_bad_good,p_bad_good = stats.ttest_rel(bad_review_test['good_revenue'],bad_review_test['Worldwide Gross'])
print(p_bad_good)
t_stat_good_bad,p_good_bad = stats.ttest_rel(good_review_test['bad_revenue'],good_review_test['Worldwide Gross'])
print(p_good_bad)

# Code for doubly robust estimator
# Create 3 training and 3 test datasets for positive and mixed, positive and negative and negative and mixed reviews
# The training data with 2 types of reviews (positive and mixed, positive and negative and negative, mixed reviews)
# will be used for propensity score
# The training data is selected from each of this training data to get a pure positve reviews training data set,
# pure negative reviews dataset and pure mixed reviews datasets
# These pure datasets are used to model the movie revenue regression model 

good_neut_review_log_data = critic_cast_revenue.loc[critic_cast_revenue['Good Critic Review']>=0,:].copy()
bad_neut_review_log_data = critic_cast_revenue.loc[critic_cast_revenue['Good Critic Review']<=0,:].copy()
good_bad_review_log_data = critic_cast_revenue.loc[(critic_cast_revenue['Good Critic Review']==-1) | (critic_cast_revenue['Good Critic Review']==1),:].copy()

bad_neut_review_log_data['Bad Critic Review'] = bad_neut_review_log_data['Good Critic Review'].replace(-1,1)
good_bad_review_log_data['Good Critic Review'].replace(-1,0,inplace=True)

good_neut_review_log_data.loc[:,'random'] = np.random.random(len(good_neut_review_log_data))
bad_neut_review_log_data.loc[:,'random'] = np.random.random(len(bad_neut_review_log_data))
good_bad_review_log_data.loc[:,'random'] = np.random.random(len(good_bad_review_log_data))

X_train_good_neut_review = good_neut_review_log_data.loc[good_neut_review_log_data['random']<0.5,:].copy()
X_test_good_neut_review = good_neut_review_log_data.loc[good_neut_review_log_data['random']>=0.5,:].copy()
y_train_good_neut_review = good_neut_review_log_data.loc[good_neut_review_log_data['random']<0.5,'Good Critic Review'].copy()
X_train_bad_neut_review = bad_neut_review_log_data.loc[bad_neut_review_log_data['random']<0.5,:].copy()
X_test_bad_neut_review = bad_neut_review_log_data.loc[bad_neut_review_log_data['random']>=0.5,:].copy()
y_train_bad_neut_review = bad_neut_review_log_data.loc[bad_neut_review_log_data['random']<0.5,'Bad Critic Review'].copy()
X_train_good_bad_review = good_bad_review_log_data.loc[good_bad_review_log_data['random']<0.5,:].copy()
X_test_good_bad_review = good_bad_review_log_data.loc[good_bad_review_log_data['random']>=0.5,:].copy()
y_train_good_bad_review = good_bad_review_log_data.loc[good_bad_review_log_data['random']<0.5,'Good Critic Review'].copy()

good_review_lr_data_1 = X_train_good_neut_review.loc[X_train_good_neut_review['Good Critic Review']==1,:].copy()
good_review_lr_data_2 = X_train_good_bad_review.loc[X_train_good_bad_review['Good Critic Review']==1,:].copy()
neut_review_lr_data_1 = X_train_good_neut_review.loc[X_train_good_neut_review['Good Critic Review']==0,:].copy()
neut_review_lr_data_2 = X_train_bad_neut_review.loc[X_train_bad_neut_review['Bad Critic Review']==0,:].copy()
bad_review_lr_data_1 = X_train_bad_neut_review.loc[X_train_bad_neut_review['Bad Critic Review']==1,:].copy()
bad_review_lr_data_2 = X_train_good_bad_review.loc[X_train_good_bad_review['Good Critic Review']==0,:].copy()

# Code for random forest regression for each of the review types (i.e., positive, mixed and negative).
# Randomized grid search based on fixed parameters was done to get the best estimator
# 2 models were made to prevent leakage 

from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(10, 50, num = 10)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(5, 25, num = 10)]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

goodreview_revenue_model_1 = RandomizedSearchCV(estimator = rf(), param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
goodreview_revenue_model_2 = RandomizedSearchCV(estimator = rf(), param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
neutreview_revenue_model_1 = RandomizedSearchCV(estimator = rf(), param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
neutreview_revenue_model_2 = RandomizedSearchCV(estimator = rf(), param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
badreview_revenue_model_1 = RandomizedSearchCV(estimator = rf(), param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
badreview_revenue_model_2 = RandomizedSearchCV(estimator = rf(), param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)

goodreview_revenue_model_1.fit(good_review_lr_data_1.loc[:,model_cols],good_review_lr_data_1['Worldwide Gross'])
goodreview_revenue_model_2.fit(good_review_lr_data_2.loc[:,model_cols],good_review_lr_data_2['Worldwide Gross'])
neutreview_revenue_model_1.fit(neut_review_lr_data_1.loc[:,model_cols],neut_review_lr_data_1['Worldwide Gross'])
neutreview_revenue_model_2.fit(neut_review_lr_data_2.loc[:,model_cols],neut_review_lr_data_2['Worldwide Gross'])
badreview_revenue_model_1.fit(bad_review_lr_data_1.loc[:,model_cols],bad_review_lr_data_1['Worldwide Gross'])
badreview_revenue_model_2.fit(bad_review_lr_data_2.loc[:,model_cols],bad_review_lr_data_2['Worldwide Gross'])

# Code for logistic regression for propensity score.

good_neut_model = LogisticRegression()
bad_neut_model = LogisticRegression()
good_bad_model = LogisticRegression()

good_neut_model.fit(X_train_good_neut_review.loc[:,model_cols],y_train_good_neut_review)
bad_neut_model.fit(X_train_bad_neut_review.loc[:,model_cols],y_train_bad_neut_review)
good_bad_model.fit(X_train_good_bad_review.loc[:,model_cols],y_train_good_bad_review)

# Predicting the propensity score for the test data
# Also predicting revenue for the test data

X_test_good_neut_review['prop_score'] = good_neut_model.predict_proba(X_test_good_neut_review.loc[:,model_cols])[:,1]
X_test_bad_neut_review['prop_score'] = bad_neut_model.predict_proba(X_test_bad_neut_review.loc[:,model_cols])[:,1]
X_test_good_bad_review['prop_score'] = good_bad_model.predict_proba(X_test_good_bad_review.loc[:,model_cols])[:,1]

X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==0,'good_rev'] = goodreview_revenue_model_1.best_estimator_ .predict(X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==0,model_cols])
X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==1,'good_rev'] = goodreview_revenue_model_1.best_estimator_ .predict(X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==1,model_cols])
X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==1,'neut_rev'] = neutreview_revenue_model_1.best_estimator_ .predict(X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==1,model_cols])
X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==0,'neut_rev'] = neutreview_revenue_model_1.best_estimator_ .predict(X_test_good_neut_review.loc[X_test_good_neut_review['Good Critic Review']==0,model_cols])

X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==0,'bad_rev'] = badreview_revenue_model_1.best_estimator_ .predict(X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==0,model_cols])
X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==1,'bad_rev'] = badreview_revenue_model_1.best_estimator_ .predict(X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==1,model_cols])
X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==0,'neut_rev'] = neutreview_revenue_model_2.best_estimator_ .predict(X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==0,model_cols])
X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==1,'neut_rev'] = neutreview_revenue_model_2.best_estimator_ .predict(X_test_bad_neut_review.loc[X_test_bad_neut_review['Bad Critic Review']==1,model_cols])

X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==1,'bad_rev'] = badreview_revenue_model_2.best_estimator_ .predict(X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==1,model_cols])
X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==0,'good_rev'] = goodreview_revenue_model_2.best_estimator_ .predict(X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==0,model_cols])
X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==0,'bad_rev'] = badreview_revenue_model_2.best_estimator_ .predict(X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==0,model_cols])
X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==1,'good_rev'] = goodreview_revenue_model_2.best_estimator_ .predict(X_test_good_bad_review.loc[X_test_good_bad_review['Good Critic Review']==1,model_cols])

# Final Doubly Robust Estimates of Positive and mixed reviews
Delta_Doubly_Robust_part1_1 = X_test_good_neut_review['Good Critic Review']*X_test_good_neut_review['Worldwide Gross']/X_test_good_neut_review['prop_score'] 
Delta_Doubly_Robust_part1_2 = (X_test_good_neut_review['Good Critic Review'] - X_test_good_neut_review['prop_score'])/X_test_good_neut_review['prop_score']*X_test_good_neut_review['good_rev'] 
Delta_Doubly_Robust_part2_1 = (1-X_test_good_neut_review['Good Critic Review'])*X_test_good_neut_review['Worldwide Gross']/(1-X_test_good_neut_review['prop_score']) 
Delta_Doubly_Robust_part2_2 = (X_test_good_neut_review['Good Critic Review'] - X_test_good_neut_review['prop_score'])/(1-X_test_good_neut_review['prop_score'])*X_test_good_neut_review['neut_rev'] 

Delta_Doubly_Robust_Estimate = np.mean(Delta_Doubly_Robust_part1_1 - Delta_Doubly_Robust_part1_2 - Delta_Doubly_Robust_part2_1 - Delta_Doubly_Robust_part2_2)
