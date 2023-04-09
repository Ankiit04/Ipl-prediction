from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    app_running = True
    signal.signal(signal.SIGINT, signal_handler)
    while app_running:
        try:
            # Run your prediction app logic here
            pass
        except Exception as e:
            # Handle exceptions if any
            print("Error occurred:", e)
            # Additional error handling code can be added here

    app.run(debug=True)



app = Flask(__name__)


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

teams_to_remove = ['Kochi Tuskers Kerala', 'Pune Warriors','Deccan Chargers']

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

@app.route('/')
def form():
    # Get unique team names and venue names
    teams = team_label_encoder.inverse_transform(ipl['team1'].unique())
    venues = venue_label_encoder.inverse_transform(ipl['venue'].unique())
    return render_template('form.html', teams=teams, venues=venues)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Access user input from form
        team1 = request.form['team1']
        team2 = request.form['team2']
        venue = request.form['venue']
        
        # Encode user input using label encoder
        team1_encoded = team_label_encoder.transform([team1])[0]
        team2_encoded = team_label_encoder.transform([team2])[0]
        venue_encoded = venue_label_encoder.transform([venue])[0]
        
        # Modify the new_data to use team1 and team2 selected by the user
        new_data = [[team1_encoded, team2_encoded, venue_encoded]]
        
        # Make prediction
        predicted_winner = model.predict(new_data)
        
        # Decode the predicted winner label
        decoded_winner = team_label_encoder.inverse_transform(predicted_winner)
        
        # Return predicted winner to the template for rendering
        return render_template('result.html', predicted_winner=decoded_winner[0])
    
    # Render the form for user input
    return render_template('form.html')
@app.route('/tipme')
def goback():
    return render_template('tipme.html')

if __name__ == '__main__':
    app.run(debug=True)
