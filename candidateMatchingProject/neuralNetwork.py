import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from transformers import pipeline
import numpy as np


# Load job postings and candidate resumes
job_postings = pd.read_csv('job_postings.csv')
candidate_resumes = pd.read_csv('candidate_resumes.csv')

# Encode labels
label_encoder = LabelEncoder()
job_postings['label'] = label_encoder.fit_transform(job_postings['label'])

# Feature Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_job_postings = tfidf_vectorizer.fit_transform(job_postings['text'])
X_resumes = tfidf_vectorizer.transform(candidate_resumes['text'])
y = job_postings['label']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resumes, y, test_size=0.2, random_state=42)

# Build Neural Network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# AI Integration (ChatGPT)
chat_gpt = pipeline("text-generation", model="gpt2")

# User Interface Development
def recommend_candidates(job_description):
    job_description_features = tfidf_vectorizer.transform([job_description])
    predicted_probabilities = model.predict(job_description_features)

    # Get the index of the candidate with the highest probability
    best_candidate_index = np.argmax(predicted_probabilities)

    # Extract the corresponding candidate ID and name
    best_candidate_id = candidate_resumes.iloc[best_candidate_index]['resume_id']
    best_candidate_name = candidate_resumes.iloc[best_candidate_index]['candidate_name']

    # Generate response using ChatGPT
    response = chat_gpt("Please find candidates matching the following job description:\n" + job_description)

    return best_candidate_name, response[0]['generated_text']

# Example usage
job_description = "Software Engineer"
best_candidate, chat_response = recommend_candidates(job_description)
print("Best candidate for job description '{}': {}".format(job_description, best_candidate))
# print("ChatGPT Response:\n", chat_response)
