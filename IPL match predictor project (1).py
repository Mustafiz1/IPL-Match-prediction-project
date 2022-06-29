#!/usr/bin/env python
# coding: utf-8

# ## IPL MATCH PREDICTION PROJECT

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


#pip install --upgrade pip

import pip
pip.main(['install', 'scikit-optimize==0.8.1'])
#pip.main(['install', 'scikit-learn==0.22.2'])
#pip install scikit-optimize==0.8.1
#pip install scikit-learn==0.22.2


# In[3]:


match = pd.read_csv('matches.csv')


# In[4]:


match.head()


# In[5]:


deliver = pd.read_csv('deliveries.csv')


# In[6]:


deliver.head()


# In[7]:


total_score_df = deliver.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()


# In[8]:


total_score_df


# In[9]:


total_score_df = total_score_df[total_score_df['inning'] == 1]


# In[10]:


match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id',right_on = 'match_id')


# In[11]:


match_df.head()


# In[12]:


match_df['team1'].unique()


# ## Current playing teams

# In[13]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[14]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')


# In[15]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[16]:


match_df.shape


# ## Removing mathches with rain

# In[17]:


matches_df = match_df[match_df['dl_applied'] == 0]


# In[18]:


matches_df.head()


# In[19]:


match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]


# In[21]:


deliver.shape


# ## Merging two dataframe on 'match_id'

# In[22]:


deliver_df = match_df.merge(deliver, on = 'match_id')


# In[23]:


deliver_df = deliver_df[deliver_df['inning'] == 2]


# In[24]:


deliver_df.shape


# In[25]:


deliver_df['current_score'] = deliver_df.groupby('match_id').cumsum()['total_runs_y']


# In[26]:


deliver_df.sample(10)


# In[27]:


deliver_df['runs_left'] = (deliver_df['total_runs_x']+1) - deliver_df['current_score']


# In[28]:


deliver_df


# In[29]:


deliver_df['balls_left'] = 126 - (deliver_df['over'] * 6 + deliver_df['ball'])


# In[30]:


deliver_df.head()


# In[31]:


deliver_df.shape


# In[32]:


deliver_df['player_dismissed'] = deliver_df['player_dismissed'].fillna("0")
deliver_df['player_dismissed'] = deliver_df['player_dismissed'].apply(lambda x:x if x=='0' else '1')
deliver_df['player_dismissed'] = deliver_df['player_dismissed'].astype('int')
wickets = deliver_df.groupby('match_id').cumsum()['player_dismissed'].values
deliver_df['wickets_left'] = 10- wickets 
deliver_df.tail()


# ## Currenr Run Rate

# In[33]:


deliver_df['crr'] = (deliver_df['current_score'] * 6/(120-deliver_df['balls_left']))


# In[34]:


deliver_df.head()


# In[35]:


deliver_df['rrr'] = (deliver_df['runs_left'] * 6)/(deliver_df['balls_left'])


# In[36]:


deliver_df.head()


# In[37]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[38]:


deliver_df['result'] = deliver_df.apply(result, axis =1)


# In[39]:


deliver_df


# In[40]:


final_df = deliver_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'total_runs_x', 'crr', 'rrr', 'result']]


# In[41]:


final_df = final_df.sample(final_df.shape[0])


# In[42]:


final_df.sample()


# ## Removing 'nan' values

# In[43]:


final_df.dropna(inplace = True)


# In[44]:


final_df


# ## Removing the last ball of the match, because it creates infinite value for 'rrr'

# In[45]:


final_df = final_df[final_df['balls_left'] != 0]


# ## Model Building

# In[46]:


X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1)


# In[49]:


X_train.shape


# In[50]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse = False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder = 'passthrough')


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# In[49]:


# pipe = Pipeline(steps= [
#     ('step1', trf),
#     ('step2', RandomForestClassifier())
# ])


# In[52]:


pipe = Pipeline(steps= [
    ('step1', trf),
    ('step2', LogisticRegression(solver ='liblinear'))
])


# In[53]:


pipe.fit(X_train,y_train)


# In[54]:


y_pred = pipe.predict(X_test)


# In[55]:


y_pred


# In[56]:


y_test


# In[57]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[63]:


pipe.predict_proba(X_test)[11]


# In[92]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
    


# In[104]:


match_summary(207)#,'Mumbai Indians','207')


# In[93]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets_left = list(temp_df['wickets_left'].values)
    new_wickets = wickets_left[:]
    new_wickets.insert(0,10)
    wickets_left.append(0)
    w = np.array(wickets_left)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
    


# In[94]:


temp_df,target = match_progression(deliver_df, 45, pipe)


# In[95]:


temp_df


# In[96]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[97]:


teams


# In[98]:


city = deliver_df['city'].unique()


# In[99]:


import pickle


# In[100]:


pickle.dump(pipe, open('pipe.pkl', 'wb'))


# In[ ]:





# In[ ]:




