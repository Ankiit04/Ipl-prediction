import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
ipl = pd.read_csv("path/to/your/ipl.csv")  # Replace with the path to your dataset

# Add 'Rising Pune Supergiant' to the list of labels
ipl['label'] = ipl['label'].append(pd.Series(['Rising Pune Supergiant']))

# Encode the labels
label_encoder = LabelEncoder()
ipl['label'] = label_encoder.fit_transform(ipl['label'])

# Split the data into training and testing sets
X = ipl.iloc[:, :-1]  # Features
y = ipl.iloc[:, -1]  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Create a Streamlit app
def predict_label(features):
    X_input = pd.DataFrame([features], columns=X.columns)
    y_pred = clf.predict(X_input)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    return y_pred_decoded[0]

def main():
    st.title("IPL Team Prediction")
    st.write("Enter the values for the features")

    # Collect input from the user
    feature1 = st.slider("Feature 1", min_value=0, max_value=10, value=5)
    feature2 = st.slider("Feature 2", min_value=0, max_value=10, value=5)
    feature3 = st.slider("Feature 3", min_value=0, max_value=10, value=5)

    # Predict on the input data
    prediction = predict_label([feature1, feature2, feature3])

    # Display the predicted label
    st.write("Predicted Label:", prediction)

if __name__ == "__main__":
    main()
