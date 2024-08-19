# import numpy as np

# # SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# # THE NAME OF THE PYTHON FILE MUST BE submit.py

# # DO NOT PERFORM ANY FILE IO IN YOUR CODE

# # DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# # IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# # CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# # You may define any new functions, variables, classes here
# # For example, classes to create the Tree, Nodes etc

# ################################
# # Non Editable Region Starting #
# ################################
# def my_fit( words ):
# ################################
# #  Non Editable Region Ending  #
# ################################

# 	# Do not perform any file IO in your code
# 	# Use this method to train your model using the word list provided
	
# 	return model					# Return the trained model


# ################################
# # Non Editable Region Starting #
# ################################
# def my_predict( model, bigram_list ):
# ################################
# #  Non Editable Region Ending  #
# ################################
	
# 	# Do not perform any file IO in your code
# 	# Use this method to predict on a test bigram_list
# 	# Ensure that you return a list even if making a single guess
	
# 	return guess_list					# Return guess(es) as a list





# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from collections import defaultdict
# import itertools

# # Generate bigrams from a word
# def generate_bigrams(word):
#     return [word[i:i+2] for i in range(len(word) - 1)]

# # Process bigrams (sorted + duplicate)
# def process_bigrams(bigrams):
#     unique_bigrams = sorted(set(bigrams))
#     return unique_bigrams[:5]

# # Prepare the dataset
# def prepare_dataset(dictionary):
#     bigram_dict = defaultdict(list)
#     for word in dictionary:
#         bigrams = generate_bigrams(word)
#         processed_bigrams = process_bigrams(bigrams)
#         bigram_dict[tuple(processed_bigrams)].append(word)
#     return bigram_dict

# # Load the dictionary
# with open('dict', 'r') as file:
#     dictionary = [line.strip() for line in file]

# # Prepare the dataset
# bigram_dict = prepare_dataset(dictionary)
# print("Bigram dictionary prepared with keys:")
# for key in bigram_dict.keys():
#     print(key)

# def my_fit(dictionary):
#     bigram_dict = prepare_dataset(dictionary)
    
#     # Create feature vectors and labels
#     X = []
#     y = []
#     bigram_to_index = {bigram: idx for idx, bigram in enumerate(sorted(set(itertools.chain(*bigram_dict.keys()))))}
    
#     for bigrams, words in bigram_dict.items():
#         feature_vector = [0] * len(bigram_to_index)
#         for bigram in bigrams:
#             feature_vector[bigram_to_index[bigram]] = 1
#         X.append(feature_vector)
        
#         # Each bigram sequence corresponds to one word
#         if words:  # Words list is not empty
#             y.append(words[0])  # Use the first word as the label
    
#     X = np.array(X)
#     y = np.array(y)
    
#     # Train the decision tree model
#     model = DecisionTreeClassifier(max_depth=10)
#     model.fit(X, y)
    
#     return model, bigram_to_index

# # Train the model
# model, bigram_to_index = my_fit(dictionary)

# def my_predict(model, bigram_to_index, bigrams):
#     processed_bigrams = process_bigrams(bigrams)
#     feature_vector = [0] * len(bigram_to_index)
#     for bigram in processed_bigrams:
#         if bigram in bigram_to_index:
#             feature_vector[bigram_to_index[bigram]] = 1
    
#     print("Feature vector for prediction:")
#     print(feature_vector)
    
#     # Predict the word
#     predictions = model.predict([feature_vector])
#     print("Prediction:")
#     print(predictions)
    
#     return predictions[0][:5]

# # Example prediction
# bigrams = ('ac', 'cc', 'ce', 'es', 'ss')
# print(my_predict(model, bigram_to_index, bigrams))




import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import itertools

######### Generate bigrams from a word
def generate_bigrams(word):
    return [word[i:i+2] for i in range(len(word) - 1)]

####### Process bigrams (sorted + remove duplicates)
def process_bigrams(bigrams):
    unique_bigrams = sorted(set(bigrams))
    return unique_bigrams[:5]

###### Prepare the dataset
def prepare_dataset(dictionary):
    bigram_dict = defaultdict(list)
    for word in dictionary:
        bigrams = generate_bigrams(word)
        processed_bigrams = process_bigrams(bigrams)
        bigram_dict[tuple(processed_bigrams)].append(word)
    return bigram_dict

# Implement the my_fit() method that takes a dictionary as a list of words and returns a trained ML model.
def my_fit(dictionary):
    bigram_dict = prepare_dataset(dictionary)
    
    X = []  # Store feature vectors
    y = []  # Store words
    
    # bigram_to_index is a dictionary that maps each unique bigram to a unique index.
    bigram_to_index = {bigram: idx for idx, bigram in enumerate(sorted(set(itertools.chain(*bigram_dict.keys()))))}
    
    # Keeping track of all possible words for each set of bigrams
    bigram_to_words = defaultdict(list)
    
    for bigrams, words in bigram_dict.items():
        feature_vector = [0] * len(bigram_to_index)
        for bigram in bigrams:
            feature_vector[bigram_to_index[bigram]] = 1
        X.append(feature_vector)
        
        # Associate this set of bigrams with all possible words
        for word in words:
            bigram_to_words[tuple(feature_vector)].append(word)
    
    X = np.array(X)
    # Use the first word of each bigram set for training
    y = np.array([words[0] for words in bigram_dict.values()])
    
    # Train the decision tree model
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X, y)
    
    return model, bigram_to_index, bigram_to_words

# Train the model
model, bigram_to_index, bigram_to_words = my_fit(dictionary)

# Implement the my_predict() method that takes your learnt model and a tuple of bigrams
# sorted in ascending order, and returns a list of guesses. The my_predict() must return a
# list even if it is making only a single guess.
def my_predict(model, bigram_to_index, bigram_to_words, bigrams):
    processed_bigrams = process_bigrams(bigrams)
    feature_vector = [0] * len(bigram_to_index)
    for bigram in processed_bigrams:
        if bigram in bigram_to_index:
            feature_vector[bigram_to_index[bigram]] = 1
    
    feature_tuple = tuple(feature_vector)
    
    # Retrieve all possible words that match this set of bigrams
    if feature_tuple in bigram_to_words:
        predictions = bigram_to_words[feature_tuple]
    else:
        # If no match in bigram_to_words, use model prediction and look up the closest matches
        prediction = model.predict([feature_vector])[0]
        prediction_bigrams = tuple(process_bigrams(generate_bigrams(prediction)))
        predictions = bigram_dict[prediction_bigrams] if prediction_bigrams in bigram_dict else [prediction]
    
    return predictions[:5]

# Load the dictionary
with open('dict_secret.txt', 'r') as file:
    dictionary = [line.strip() for line in file]

# Train the model
model, bigram_to_index, bigram_to_words = my_fit(dictionary)

# Example predictions
test_bigrams = [
    ('al','io','na','on','op'),
    ('ar','ca','rr','ry'),
    ('ar','re','ea'),
    ('ci','en','ff','fi','ic'),
    ('be','de','es','id','si')
]

# Predict for each set of bigrams and print the results
for bigrams in test_bigrams:
    print(f"Bigrams: {bigrams}")
    predictions = my_predict(model, bigram_to_index, bigram_to_words, bigrams)
    print(f"Predictions: {predictions}\n")