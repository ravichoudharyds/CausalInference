
# coding: utf-8

import numpy as np
import pandas as pd
import json
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


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

critic_revenue["log Worldwide Gross"]=np.log(critic_revenue["Worldwide Gross"])#adding column for log of revenue

#Normalizing by the median for merged data set

critic_revenue["Production Budget"]=(critic_revenue["Production Budget"]-critic_revenue["Production Budget"].median())/critic_revenue["Production Budget"].std()
critic_revenue["Worldwide Gross"]=(critic_revenue["Worldwide Gross"]-critic_revenue["Worldwide Gross"].median())/critic_revenue["Worldwide Gross"].std()
critic_revenue["log Worldwide Gross"]=(critic_revenue["log Worldwide Gross"]-critic_revenue["log Worldwide Gross"].median())/critic_revenue["log Worldwide Gross"].std()
critic_revenue["Domestic Gross"]=(critic_revenue["Domestic Gross"]-critic_revenue["Domestic Gross"].median())/critic_revenue["Domestic Gross"].std()
critic_revenue["popularity"]=(critic_revenue["popularity"]-critic_revenue["popularity"].median())/critic_revenue["popularity"].std()
critic_revenue["vote_count"]=(critic_revenue["vote_count"]-critic_revenue["vote_count"].median())/critic_revenue["vote_count"].std()
critic_revenue["metacritic_metascore"]=(critic_revenue["metacritic_metascore"]-critic_revenue["metacritic_metascore"].median())/critic_revenue["metacritic_metascore"].std()
critic_revenue["runtime"]=(critic_revenue["runtime"]-critic_revenue["runtime"].median())/critic_revenue["runtime"].std()

# Reindex since the dataset has been subset many times
critic_revenue.index=range(len(critic_revenue))

# Goal is to create dummy variables from different columns with json string

for movie_num in range(len(critic_revenue)):
    flat_json=json.loads(critic_revenue.loc[movie_num,'production_companies'])
    for index_json in range(len(flat_json)):
        critic_revenue.loc[movie_num,flat_json[index_json]['name']] = 1

for movie_num in range(len(critic_revenue)):
    flat_json=json.loads(critic_revenue.loc[movie_num,'genres'])
    for index_json in range(len(flat_json)):
        critic_revenue.loc[movie_num,flat_json[index_json]['name']] = 1

# Goal: To reduce the number of production companies from the columns
# Get Description of data
data_descr = critic_revenue.describe()

# Variable to store count of movies for each production Companies
movie_count=[]

# Variable to store names of Production Companies
production_house=[]

#These are the columns we want to keep
keep_list=['Unnamed: 0','Unnamed: 0.1','id','genres','popularity',
               'runtime','vote_average','vote_count','metacritic_metascore','metacritic_metascore1','Action',
               'Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family',
               'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War', 'Music', 'Documentary',
               'Foreign', 'TV Movie','production_companies','production_countries','release_date','spoken_languages',
              'title','original_title','original_language','release_year','Release Date','Production Budget','Domestic Gross','Worldwide Gross','num_year','log Worldwide Gross']

# Movie Count for each production company is maintained
for column in data_descr.columns:
    if column not in keep_list:
        movie_count.append(data_descr.loc['count',column])

# Keep Production Company name if they have produced more than 99th percentile of movie_count 
for column in data_descr.columns:
    if column not in keep_list:
        if data_descr.loc['count',column] > np.percentile(movie_count,99):
            production_house.append(column)

# Also keep the list from Wikipedia: Major Studio Houses and Mini-Major Studio Houses of past and present
production_house.extend(['NBCUniversal','Comcast','Universal Pictures','Focus Features','Gramercy Pictures', 'Working Title Films', 'Big Idea',
'Bullwinkle Studios', 'DreamWorks Animation', 'Illumination', 'Illumination Mac Guff',
'Universal Animation Studios', 'Amblin Partners', 'Focus World', 'High Top Releasing', 'NBCU Ent. Japan',
'Viacom','National Amusements','Paramount Pictures','Insurge Pictures','Paramount Famous Productions'
,'Paramount Players','MTV Animation','Nickelodeon Animation Studio','Paramount Animation','Awesomeness Films'
,'BET Films','CMT Films','Comedy Central Films','MTV Films','Nickelodeon Movies','VH1 Films','Viacom 18 Motion Pictures'
,'Sony Pictures','Sony','Columbia Pictures','Sony Pictures Classics','Screen Gems','TriStar Pictures'
,'Funimation Films','Sony Pictures Animation','Sony Pictures Imageworks','Affirm Films','Destination Films'
,'Left Bank Pictures','Sony Pictures Releasing','SPWA','Stage 6 Films','TriStar Productions'
,'Triumph Films','WarnerMedia','AT&T','Warner Bros. Pictures','DC Films','New Line Cinema'
,'Cartoon Network Studios','Hanna-Barbera','Warner Animation Group','Warner Bros. Animation'
,'Adult Swim Films','Castle Rock Entertainment','Cinemax Films','CNN Films','Flagship Entertainment'
,'HBO Films','Machinima, Inc.','Turner Entertainment','Williams Street','Walt Disney Studios'
,'The Walt Disney Company','Walt Disney Pictures','Disneynature','Lucasfilm','Marvel Studios','The Muppets Studio'
,'Lucasfilm Animation','Marvel Animation','Pixar Animation Studios','Walt Disney Animation Studios'
,'A&E IndieFilms','ESPN Films','Miravista Films','VICE Films','Walt Disney Studios Motion Pictures'
,'Fox Entertainment Group','21st Century Fox','20th Century Fox' ,'Fox Searchlight Pictures','Fox 2000 Pictures'
,'Regency Enterprises','20th Century Fox Animation','Blue Sky Studios','Fox Star Studios'
,'Kudos Film and Television','New Regency','Tiger Aspect Productions','Zero Day Fox','United Artists'
,'RKO Pictures','Metro-Goldwyn-Mayer Pictures''Lionsgate Motion Picture Group','Lionsgate','Lionsgate Films',
'CodeBlack Films','Globalgate Entertainment','Good Universe','Lionsgate Premiere','Manga Entertainment',
'Pantelion Films','Roadside Attractions','Starz Digital Media','Starz Distribution','Summit Entertainment',
'The Amblin Group','Participant Media','Reliance Entertainment','Entertainment One','Alibaba Pictures',
'Universal Pictures','Amblin Partners','Amblin Entertainment','DreamWorks Pictures','STX Entertainment',
'Hony Capital','Tencent','PCCW','TPG Growth','Liberty Global','STXfilms','STXinternational','STXfamily',
'Gaumont Film Company','Gaumont Animation','CBS Corporation','National Amusements','CBS Films','MGM Holdings ',
'Metro-Goldwyn-Mayer','MGM Animation','United Artists','Orion Pictures','Orion Classics','Mirror','Egmont Group',
'Nordisk Film','Avanti Film','Danish Films','Maipo Film','Min Bio','Solar Films','Trust Nordisk','Zentropa(JV)',
'Constantin Film ','Hager Moss Film'])

# If the movie is not in the above lists then assume it's from an independent production house, i.e., make a dummy variable 
# named independent production
critic_revenue['Independent_Production']=0
independent_list = [column for column in critic_revenue.columns if column not in production_house and column not in keep_list]

for i in range(len(critic_revenue)):
    for company_name in independent_list:
        if critic_revenue.loc[i,company_name] == 1:
            critic_revenue.loc[i,'Independent_Production'] = 1
            break

# Drop the Independent Production House list from dataframe to reduce size
critic_revenue.drop(columns=independent_list,inplace=True)

# Need to fill the NaN/Missing values with 0 for dummy variable
Genres = ['Action','Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family',
               'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War', 'Music', 'Documentary',
               'Foreign', 'TV Movie']
prod_comp_columns = [column for column in critic_revenue.columns if column in production_house ]
critic_revenue[prod_comp_columns].fillna(0,inplace=True,axis=1)
critic_revenue[Genres].fillna(0,inplace=True,axis=1)

# Create features indicating release on a US Federal Holiday or on the weekend, also a feature indicate the day of week of release
critic_revenue['release_datetime'] = pd.to_datetime(critic_revenue['release_date'],format='%Y-%m-%d')
holiday_list = calendar().holidays(start=critic_revenue['release_datetime'].min(),end=critic_revenue['release_datetime'].max())

critic_revenue['release_date_long_weekend'] = critic_revenue['release_datetime'].isin(holiday_list)
critic_revenue['release_day_of_week'] = critic_revenue['release_datetime'].dt.weekday

for i in range(len(critic_revenue)):
    if critic_revenue.loc[i,'release_day_of_week'] ==5 or  critic_revenue.loc[i,'release_day_of_week']== 6:
        critic_revenue.loc[i,'release_date_weekend']=1
    else:
        critic_revenue.loc[i,'release_date_weekend']=0

#Pearson Correlation
print("Pearson Correlation: ",pearsonr(critic_revenue["metacritic_metascore"],critic_revenue["Worldwide Gross"]))
print("Pearson Correlation (log): ",pearsonr(critic_revenue["metacritic_metascore"],critic_revenue["log Worldwide Gross"]))
        
#Random Forest Regressor
critic_revenue=critic_revenue.drop(["release_datetime", "production_countries","genres","production_companies"], 1)
critic_revenue = pd.get_dummies(critic_revenue)
critic_revenue= critic_revenue.fillna(0)
critic_revenue['Production Budget']=critic_revenue['Production Budget'].astype('float32', copy=False)
critic_revenue['popularity']=critic_revenue['popularity'].astype('float32', copy=False)
critic_revenue['Worldwide Gross']=critic_revenue['Worldwide Gross'].astype('float32', copy=False)
critic_revenue['vote_average']=critic_revenue['vote_average'].astype('float32', copy=False)
critic_revenue['vote_count']=critic_revenue['vote_count'].astype('float32', copy=False)
critic_revenue['metacritic_metascore']=critic_revenue['metacritic_metascore'].astype('float32', copy=False)
labels = np.array(critic_revenue['Worldwide Gross'])

# Remove the labels from the features
# axis 1 refers to the columns
critic_revenue= critic_revenue.drop('Worldwide Gross', axis = 1)


# Convert to numpy array
critic_revenue = np.array(critic_revenue)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(critic_revenue, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


#Linear Regression for Genre (note: still need to integrate with existing code as far as train/test split

#Entire dataset, metacritic metascore only
X=np.array(critic_revenue["metacritic_metascore"]).reshape(-1,1)
y=np.array(critic_revenue["log Worldwide Gross"]).reshape(-1,1)
critic_only=LinearRegression().fit(X,y)
print("Coefficient: ",critic_only.coef_)
print("Intercept: ",critic_only.intercept_ )

#plotting datapoints and line for regression
plt.scatter(X,y,marker=".")
plt.plot(X,critic_only.coef_*X+critic_only.intercept_,"r")
plt.show()

#Entire dataset, genre data only
GenresX=[Genres+["metacritic_metascore"]][0]
XX=critic_revenue[GenresX]
genre_reg=LinearRegression().fit(XX,y)
print("Coefficient: ",genre_reg.coef_)
print("Intercept: ",genre_reg.intercept_ )

#Repeating, but with different training and testing
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(critic_revenue.drop("log Worldwide Gross",1), critic_revenue["log Worldwide Gross"], test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

metacritic_trainer=LinearRegression().fit(np.array(train_features["metacritic_metascore"]).reshape(-1, 1),train_labels)
print("Coefficient: ",metacritic_trainer.coef_)
print("Intercept: ",metacritic_trainer.intercept_ )
metacritic_predict=metacritic_trainer.predict(np.array(test_features["metacritic_metascore"]).reshape(-1,1))

plt.scatter(test_features["metacritic_metascore"],test_labels,marker=".")
plt.plot(test_features["metacritic_metascore"],metacritic_trainer.coef_*test_features["metacritic_metascore"]+metacritic_trainer.intercept_,"r")
plt.plot(test_features["metacritic_metascore"],metacritic_predict,"g")
plt.show()

genres_trainer=LinearRegression().fit(np.array(train_features[GenresX]),train_labels)
print("Coefficient: ",genres_trainer.coef_)
print("Intercept: ",genres_trainer.intercept_ )
genres_predict=genres_trainer.predict(np.array(test_features[GenresX]))
