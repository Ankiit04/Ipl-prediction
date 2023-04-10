import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the IPL matches dataset
ipl = pd.read_csv('matches.csv')

# Define dictionary to rename team names
team_rename_dict = {
    'Gujarat Lions': 'Gujarat Titans',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Lucknow Supergiants',
    'Delhi Daredevils': 'Delhi Capitals',
    # Add more team name mappings as needed
}

# Define dictionary to rename venue names
venue_rename_dict = {
    'ACA-VDCA Stadium': 'Arun Jaitley Stadium',
    'Green Park': 'Narendra Modi Stadium',
    'Buffalo Park': 'Ekana Cricket Stadium',
    # Add more venue name mappings as needed
}

# Rename team names
ipl['team1'] = ipl['team1'].replace(team_rename_dict)
ipl['team2'] = ipl['team2'].replace(team_rename_dict)
ipl['winner'] = ipl['winner'].replace(team_rename_dict)

# Rename venue names
ipl['venue'] = ipl['venue'].replace(venue_rename_dict)

# Drop unnecessary columns
ipl.drop(['dl_applied', 'toss_winner', 'toss_decision', 'result', 'win_by_runs', 'win_by_wickets', 'player_of_match', 'umpire1', 'umpire2', 'umpire3', 'date', 'city', 'season', 'id'], axis=1, inplace=True)

# Drop rows with NaN values in the 'winner' column
ipl.dropna(subset=['winner'], inplace=True)

# Encode categorical variables
team_label_encoder = LabelEncoder()
venue_label_encoder = LabelEncoder()
ipl['team1'] = team_label_encoder.fit_transform(ipl['team1'])
ipl['team2'] = team_label_encoder.transform(ipl['team2'])
ipl['winner'] = team_label_encoder.transform(ipl['winner'])
ipl['venue'] = venue_label_encoder.fit_transform(ipl['venue'])

teams_to_remove = ['Kochi Tuskers Kerala', 'Pune Warriors', 'Deccan Chargers']

# Filter out matches involving teams to be removed
ipl = ipl[~ipl['team1'].isin(teams_to_remove)]
ipl = ipl[~ipl['team2'].isin(teams_to_remove)]
ipl = ipl[~ipl['winner'].isin(teams_to_remove)]

# Get unique team names and venue names
teams = [team for team in team_label_encoder.inverse_transform(ipl['team1'].unique())]
teams.remove(teams_to_remove[0])
teams.remove(teams_to_remove[1])
teams.remove(teams_to_remove[2])

venues = venue_label_encoder.inverse_transform(ipl['venue'].unique())

# Split the data into training and testing sets
X_train = ipl[['team1', 'team2', 'venue']]
y_train = ipl['winner']

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.title('IPL Match Winner Prediction')
st.markdown('---')

# Render the form for user input
team1 = st.selectbox('Select Team 1', teams)
team2 = st.selectbox('Select Team 2', teams)
venue = st.selectbox('Select Venue', venues)

if st.button('Predict'):
    # Encode user input using label encoder
    team1_encoded = team_label_encoder.transform([team1])[0]
    team2_encoded = team_label
# Make prediction
prediction = model.predict([[team1_encoded, team2_encoded, venue_encoded]])
predicted_winner = team_label_encoder.inverse_transform(prediction)[0]

# Render prediction result
st.markdown('**Predicted Winner:** ' + predicted_winner)
