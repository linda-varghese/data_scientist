import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load job postings and candidate resumes
job_postings = pd.read_csv('C:/Users/15717/PycharmProjects/candidateMatchingProject/job_postings.csv')
candidate_resumes = pd.read_csv('C:/Users/15717/PycharmProjects/candidateMatchingProject/candidate_resumes.csv')

# Feature Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_job_postings = tfidf_vectorizer.fit_transform(job_postings['text'])
X_resumes = tfidf_vectorizer.transform(candidate_resumes['text'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(candidate_resumes['resume_id'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resumes, y, test_size=0.2, random_state=42)

# Model Building (Neural Network)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# User Interface Development
def recommend_candidates(job_description):
    job_description_features = tfidf_vectorizer.transform([job_description])
    predicted_probs = model.predict(job_description_features)
    predicted_candidate_id = np.argmax(predicted_probs)
    recommended_candidate = label_encoder.inverse_transform([predicted_candidate_id])[0]
    return recommended_candidate

# Example usage
job_description = "python"
recommended_candidate = recommend_candidates(job_description)
print("Recommended candidate for job description '{}' is: {}".format(job_description, recommended_candidate))
recommended_candidate_info = candidate_resumes[candidate_resumes['resume_id'] == recommended_candidate]

# Print corresponding candidate information
print("Recommended Candidate Information:")
print(recommended_candidate_info)
