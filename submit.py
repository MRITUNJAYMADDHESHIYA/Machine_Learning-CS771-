from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pickle

def my_fit(words):
    def get_bigrams(word, lim=5):
        # Generate bigrams and sort them
        bigrams = sorted(set(''.join(x) for x in zip(word, word[1:])))
        return tuple(bigrams[:lim])  # Limit to at most `lim` bigrams
    
    # Process each word to get its bigrams
    bigrams = [get_bigrams(word) for word in words]
    
    # Collect all unique bigrams across all words
    all_bigrams = sorted(set(bigram for sublist in bigrams for bigram in sublist))
    
    # Initialize the vectorizer
    vectorizer = DictVectorizer(sparse=False)
    
    # Transform bigrams into feature vectors
    X = vectorizer.fit_transform([{bigram: (bigram in word_bigrams) for bigram in all_bigrams} for word_bigrams in bigrams])
    
    # Set y to be the actual words (labels)
    y = np.array(words)
    
    # Initialize and train the model
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X, y)
    
    # Save the vectorizer
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
    
    # Save the model classes
    with open('classes.pkl', 'wb') as classes_file:
        pickle.dump(model.classes_, classes_file)
    
    return model

import pickle
import numpy as np

def my_predict(model, bigrams):
    # Load the vectorizer from the file
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    
    # Create a feature vector from the bigrams
    def create_feature_vector(bigrams, all_bigrams):
        # Ensure all bigrams in the feature vector are in lexical order and set to 1 if present
        return {bigram: (bigram in bigrams) for bigram in all_bigrams}
    
    # Retrieve the vocabulary from the loaded vectorizer
    all_bigrams = vectorizer.get_feature_names_out()
    
    # Transform the bigrams into a feature vector
    X_test = vectorizer.transform([create_feature_vector(bigrams, all_bigrams)])
    
    # Predict using the model
    prediction_probs = model.predict_proba(X_test)
    
    # Get the top indices of the predictions
    top_indices = np.argsort(prediction_probs[0])[-1:]  # Get indices of top 5 probabilities
    
    # Ensure the indices are in the correct order
    top_indices = top_indices[::-1]  # Reverse to get highest probabilities first
    
    # Map indices to class labels (words)
    top_guesses = [model.classes_[i] for i in top_indices]
    
    return top_guesses