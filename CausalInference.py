import numpy as np
import pandas as pd
from scipy import stats
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor as knn_reg
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import RandomizedSearchCV

# Get the top 4 actors/cast of the movie and the writer and director of the movie

credits=pd.read_csv("tmdb_5000_credits.csv")
for movie_num in range(len(credits)):
    print(movie_num)
    crew_num=0
    cast_json = json.loads(credits.loc[movie_num,'cast'])
    crew_json = json.loads(credits.loc[movie_num,'crew'])
    for index_json in range(min(4,len(cast_json))):
        credits.loc[movie_num,cast_json[index_json]['name']] = 1
    for crew in crew_json:
        if crew_num == 2:
            break
        elif crew['job'].lower().strip()== 'director':
            credits.loc[movie_num,crew['name']] = 1
            crew_num=crew_num+1
        elif crew['job'].lower().strip()== 'writer':
            credits.loc[movie_num,crew['name']] = 1
            crew_num=crew_num+1

for column in credits.columns:
    credits[column].fillna(0,inplace=True)

credits.drop(columns=['cast','crew','title'],inplace=True)
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

critic_cast_revenue=critic_revenue.merge(credits,left_on ="id",right_on="movie_id",how="left")

model_cols = [column for column in critic_cast_revenue.columns if column not in keep_list]

# Need to fill the NaN/Missing values with 0 for dummy variable
for column in model_cols:
    critic_cast_revenue[column].fillna(0,inplace=True)

# The positive, mixed and negative review tags are based on the metascore description given by metacritic
critic_cast_revenue['Good Critic Review'] = [1 if critic_cast_revenue.loc[i,'metacritic_metascore'] > 60 else 0 if critic_cast_revenue.loc[i,'metacritic_metascore'] > 40 else -1 for i in range(len(critic_cast_revenue)) ]

#Normalizing Production Budget, Workdwide Gross, and applying log to get their scale
critic_cast_revenue["Log Production Budget"] = np.log(critic_cast_revenue["Production Budget"])
critic_cast_revenue["Log Worldwide Gross"] = np.log(critic_cast_revenue["Worldwide Gross"])
critic_cast_revenue["Log Production Budget Norm"]=(critic_cast_revenue["Log Production Budget"]-critic_cast_revenue["Log Production Budget"].mean())/critic_cast_revenue["Log Production Budget"].std()

model_cols.append('Log Production Budget Norm')

# Proposed Class for Propensity Score Matching. Takes care of 70-30 train-test data split,
# Propensity Score calculation through Logistic Regression, and grid search based on C values passed to the class
# As propensity score should reflect the true probability distribution, negative log loss is used as scoring metric
# kNN was used with 1 neighbor match for matching, currently matches every point based on propensity score 
# Future suggestion - use a max critical distance to get matches
# Paired T tests were used to get p values for the difference in effect of treatment for both treated and control 
# Initialize the class and use ret_significance function to get p value based on paired t-test for both treated and control groups

class Prop_Score_Matching:
    
    def __init__(self, data, target, C_values, columns):
        self.review_data = data
        self.target = target
        self.C_values = C_values
        self.model_cols = columns
        
    def dividing(self):
        self.review_data.loc[:,'random'] = np.random.random(len(self.review_data))
        train_log=self.review_data.loc[self.review_data['random']<=0.7,:].copy()
        test_log = self.review_data.loc[self.review_data['random']>0.7,:].copy()
        return test_log,train_log

    def propensityScore(self,train_log,test_log):
        log_model = GridSearchCV(LogisticRegression(), param_grid={'C':self.C_values},scoring='neg_log_loss',cv=5)
        log_model.fit(train_log.loc[:,self.model_cols],train_log[self.target])
        test_log.loc[:,'prop_score'] = log_model.predict_proba(test_log.loc[:,self.model_cols])[:,1]
        
    def kNNMatch(self, test_log): 
        a_review_test=test_log.loc[test_log[self.target]==1,:].copy()
        b_review_test=test_log.loc[test_log[self.target]==0,:].copy()
        knn_model = knn_reg(n_neighbors=1)
        knn_model.fit(np.array(a_review_test['prop_score']).reshape(-1,1),a_review_test.loc[:,"Log Worldwide Gross"])
        b_review_test.loc[:,"y_1"] = knn_model.predict(np.array(b_review_test.loc[:,'prop_score']).reshape(-1,1))
        knn_model.fit(np.array(b_review_test.loc[:,'prop_score']).reshape(-1,1),b_review_test.loc[:,"Log Worldwide Gross"])
        a_review_test.loc[:,"y_0"] = knn_model.predict(np.array(a_review_test.loc[:,'prop_score']).reshape(-1,1))
        return a_review_test, b_review_test

    def paired_t_test(self, a_review_test,b_review_test): 
        t_stat_a_b,p_a_b = stats.ttest_rel(a_review_test["y_0"],a_review_test['Log Worldwide Gross'])
        t_stat_b_a,p_b_a = stats.ttest_rel(b_review_test["y_1"],b_review_test['Log Worldwide Gross'])
        return p_a_b, p_b_a
    
    def ret_significance(self):
        print("Splitting Train and Test")
        test_data, train_data = self.dividing()
        print("Split Data \nFitting best propensity score model")
        self.propensityScore(train_data,test_data)
        print("Fit the best model \nFitting KNN")
        a_review_test, b_review_test = self.kNNMatch(test_data)
        print("Fit the kNN \nRunning T-tests")
        return self.paired_t_test(a_review_test,b_review_test)

# Proposed Class for Doubly Robust Estimator. Takes care of 70-30 train-test data split,
# Propensity Score calculation through Logistic Regression, and grid search based on C values passed to the class
# As propensity score should reflect the true probability distribution, negative log loss is used as scoring metric
# Ridge Regression was used for movie revenue calculation as the number of variables were much more than training samples 
# And it performed better than Random Forest and Decision Tree on data based on MSE
# Returns a doubly robust estimate of Average Treatment Effect

class Doubly_Robust_Estimator:
    
    def __init__(self, data, target, rf_param_grid, C_values,columns):
        self.review_data = data
        self.target = target
        self.rf_param_grid = rf_param_grid
        self.model_cols = columns
        self.C_values = C_values
        
    def train_test_split(self):
        self.review_data.loc[:,'random'] = np.random.random(len(self.review_data))
        train_data = self.review_data.loc[self.review_data['random']<=0.7,:].copy()
        test_data = self.review_data.loc[self.review_data['random']>0.7,:].copy()
        return train_data,test_data

    def propensityScore(self, train_data, test_data):
        log_model = GridSearchCV(LogisticRegression(), param_grid={'C':self.C_values},scoring='neg_log_loss',cv=5)
        log_model.fit(train_data.loc[:,self.model_cols],train_data[self.target])
        test_data.loc[:,'prop_score'] = log_model.predict_proba(test_data.loc[:,self.model_cols])[:,1]
        
    def Ridge_Reg_Estimator(self, train_data, test_data, target_y):
        train_target_y = train_data.loc[train_data[self.target]==target_y,:].copy()
        Ridge_Search = GridSearchCV(estimator = Ridge(), param_grid = {'alpha':C_range1}, cv=5, scoring='neg_mean_squared_error', n_jobs = -1)
        Ridge_Search.fit(train_target_y.loc[:,model_cols],train_target_y['Log Worldwide Gross'])
        target_var = 'y_'+str(target_y)
        test_data[target_var] = Ridge_Search.best_estimator_.predict(test_data.loc[:,self.model_cols])

    def final_estimator(self,test_data): 
        Delta_Doubly_Robust_part1_1 = test_data[self.target]*test_data['Log Worldwide Gross']/test_data['prop_score'] 
        Delta_Doubly_Robust_part1_2 = (test_data[self.target] - test_data['prop_score'])/test_data['prop_score']*test_data['y_1'] 
        Delta_Doubly_Robust_part2_1 = (1-test_data[self.target])*test_data['Log Worldwide Gross']/(1-test_data['prop_score']) 
        Delta_Doubly_Robust_part2_2 = (test_data[self.target] - test_data['prop_score'])/(1-test_data['prop_score'])*test_data['y_0'] 
        Delta_Doubly_Robust_Estimate = np.mean(Delta_Doubly_Robust_part1_1 - Delta_Doubly_Robust_part1_2 - Delta_Doubly_Robust_part2_1 - Delta_Doubly_Robust_part2_2)
        return Delta_Doubly_Robust_Estimate
    
    def return_estimate(self):
        print("Splitting Train and Test")
        test_data, train_data = self.train_test_split()
        print("Data splitting done \nFitting Logistic model")
        self.propensityScore(train_data,test_data)
        print("Logistic Model done \nFitting RF models")
        self.Ridge_Reg_Estimator(train_data, test_data, 1)
        self.Ridge_Reg_Estimator(train_data, test_data, 0)
        print("Fit the Regression Models \nCalculating Final Estimator")
        return self.final_estimator(test_data)
        
        
