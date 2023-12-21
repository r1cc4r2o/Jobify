# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

# import actions.utils.utils as utils
# import actions.utils.config as cfg
KB_CANDIDATES_PROFILES_PATH = "data\KB_user_profiles.csv.gz"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, cosine_similarity, sigmoid_kernel, euclidean_distances, manhattan_distances, cosine_distances

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import pandas as pd

class ActionProvideCandidateProfile(Action):
    def __init__(self):
        super().__init__()
        
        self.KB_CANDIDATES_PROFILES = pd.read_csv(KB_CANDIDATES_PROFILES_PATH, compression='gzip')
        
        # Text Preprocessing
        self.vectorizer = TfidfVectorizer(stop_words='english') 
        self.tfidf_matrix = self.vectorizer.fit_transform(self.KB_CANDIDATES_PROFILES['description'].fillna(''))

    def name(self) -> Text:
        return "action_provide_candidate_profile"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]: 
        
        # Get last message
        last_message = tracker.latest_message.get('text', '')
        
        # Feature Extraction
        input_vector = self.vectorizer.transform([last_message])
        
        # Similarity Calculation
        cosine_similarities = linear_kernel(input_vector, self.tfidf_matrix).flatten()   
        
        db = self.KB_CANDIDATES_PROFILES.copy()
        
        # Ranking
        db['similarity'] = cosine_similarities
        ranked_profiles = db.sort_values(by='similarity', ascending=False)
        
        db = ranked_profiles[['name', 'surname',  'salary', 'description', 'similarity']].head(1)
        
        # Format the message
        last_message = f"Name: {db['name'].values[0]} {db['surname'].values[0]} \n" \
                       f"Salary: {db['salary'].values[0]} \n" \
                       f"Description: {db['description'].values[0]} \n" \
                       f"Similarity: {db['similarity'].values[0]} \n"
                       
        # Send the message back to the user
        dispatcher.utter_message(text=last_message)
        
        return []