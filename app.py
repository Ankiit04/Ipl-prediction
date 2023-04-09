import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import signal

# Load the IPL matches dataset
ipl = pd.read_csv('matches.csv')

# Define dictionary to rename team names
team_rename_dict = {
    'Gujarat Lions': 'Gujarat Titans',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Lucknow Supergiants',
    'Delhi Daredevils': 'Delhi Capitals'
}

# Replace team names in the dataset using the team_rename_dict
ipl.replace({'team1': team_rename_dict, 'team2': team_rename_dict, 'winner': team_rename_dict}, inplace=True)

# Label encode the categorical features
le = LabelEncoder()
ipl['team1'] = le.fit_transform(ipl['team1'])
ipl['team2'] = le.transform(ipl['team2'])
ipl['toss_winner'] = le.transform(ipl['toss_winner'])
ipl['venue'] = le.transform(ipl['venue'])
ipl['winner'] = le.transform(ipl['winner'])

# Define X and y for training
X = ipl[['team1', 'team2', 'toss_winner', 'venue']]
y = ipl['winner']

# Initialize the Decision Tree Classifier
dtc = DecisionTreeClassifier()

# Fit the model
dtc.fit(X, y)

# Define signal handler to handle SIGINT (Ctrl+C)
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    exit(0)

# Register Flask routes
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    team1 = le.transform([request.form['team1']])[0]
    team2 = le.transform([request.form['team2']])[0]
    toss_winner = le.transform([request.form['toss_winner']])[0]
    venue = le.transform([request.form['venue']])[0]
    features = np.array([team1, team2, toss_winner, venue]).reshape(1, -1)
    prediction = dtc.predict(features)
    predicted_winner = le.inverse_transform(prediction)[0]
    return render_template('index.html', prediction=predicted_winner)

@app.route('/tipme')
def tipme():
    return render_template('tipme.html')

# Run the Flask app with SocketIO
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    socketio.run(app)
