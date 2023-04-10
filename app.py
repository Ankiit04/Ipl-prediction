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
    'Green Park': 'Narendra MOdi Stadium',
    'Buffalo Park' :'Ekana Cricket Stadium'

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

venues = venue_label_encoder.inverse_transform(ipl['venue'].unique())

# Split the data into training and testing sets
X_train = ipl[['team1', 'team2', 'venue']]
y_train = ipl['winner']


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Define Streamlit app
def app():
    # Set app title
    st.set_page_config(page_title="IPL Match Winner Prediction")

    # Render form for user input
    st.markdown("# IPL Match Winner Prediction")
    team1 = st.selectbox("Select team 1", team_label_encoder.inverse_transform(ipl['team1'].unique()))
    team2 = st.selectbox("Select team 2", team_label_encoder.inverse_transform(ipl['team2'].unique()))
    venue = st.selectbox("Select venue", venue_label_encoder.inverse_transform(ipl['venue'].unique()))

    # Make prediction on user input
    if st.button("Predict"):
        team1_encoded = team_label_encoder.transform([team1])[0]
        team2_encoded = team_label_encoder.transform([team2])[0]
        venue_encoded = venue_label_encoder.transform([venue])[0]
        new_data = [[team1_encoded, team2_encoded, venue_encoded]]
        predicted_winner = model.predict(new_data)
        decoded_winner = team_label_encoder.inverse_transform(predicted_winner)
        st.markdown(f"## Predicted Winner: {decoded_winner[0]}")
         # Display predicted winner
        st.success(decoded_winner[0])

        # Render table with IPL match data
        st.markdown("## IPL Match Data")
        st.write(ipl)

        # Render table with venue mapping data
        st.markdown("## Venue Mapping Data")
        venue_mapping = pd.DataFrame({'Venue ID': ipl['venue'].unique(), 'Venue Name': venue_label_encoder.inverse_transform(ipl['venue'].unique())})
        st.write(venue_mapping)

        # Render table with team mapping data
        st.markdown("## Team Mapping Data")
        team_mapping = pd.DataFrame({'Team ID': ipl['team1'].unique(), 'Team Name': team_label_encoder.inverse_transform(ipl['team1'].unique())})
        st.write(team_mapping)
if __name__ == 'app':
    app()
