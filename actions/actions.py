KB_CANDIDATES_PROFILES_PATH = "data/KB_user_profiles.csv.gz"
KB_JOBS_PATH = "data/job_descriptions.csv"

########################################################################################
########################################################################################


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, cosine_similarity, sigmoid_kernel, euclidean_distances, manhattan_distances, cosine_distances

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import pandas as pd


# GLOBAL VARIABLES set
DESCRIPTION_USERS = ''




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
        
        db = ranked_profiles[['name', 'surname',  'salary', 'description', 'skills', 'mansion', 'city', 'relocation', 'part_full_time', 'experience_years', 'level_education', 'similarity']].head(3)
        
        levels_education = ["Bachelor's Degree" if db['level_education'].values[i] == 1 else "Master's Degree" for i in range(len(db))]
        relocations = ['Yes' if db['relocation'].values[i] == 1 else 'No' for i in range(len(db))]
        
        # Format the message
        last_messages = [
                        f"Name: {db['name'].values[i]} {db['surname'].values[i]} \n" +
                        f"City: {db['city'].values[i]} \n" +
                        f"Relocation: {relocations[i]} \n" +
                        f"Salary: {db['salary'].values[i]} \n" +
                        f"Skills: {db['skills'].values[i]} \n" +
                        f"Mansion: {db['mansion'].values[i]} \n" +
                        f"Part/Full Time: {db['part_full_time'].values[i]} \n" +
                        f"Experience Years: {db['experience_years'].values[i]} \n" + 
                        f"Level Education: {levels_education[i]} \n" +
                        f"Description: {db['description'].values[i]} \n" for i in range(len(db))
                    ] 
                    #    f"Similarity: {db['similarity'].values[i]} \n"
                    
        # join the messages
        last_message = "\n\n".join(last_messages)
        
        # update the description users global variable
        global DESCRIPTION_USERS
        DESCRIPTION_USERS = {i: desc for i, desc in zip(['1', '2', '3', 'one', 'two', 'three'], last_messages + last_messages)}
        
        # Send the message back to the user
        dispatcher.utter_message(text=last_message)
        
        return []
    
class ActionCheckConfidence(Action):
    def name(self) -> Text:
        return "action_check_confidence"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # get name of the intent
        # intent = tracker.latest_message['intent'].get('name')
        
        # Get the confidence of the predicted intent
        confidence = tracker.latest_message['intent']['confidence']

        # Set a confidence threshold (you can adjust this value)
        confidence_threshold = 0.7

        # Check if the confidence is below the threshold
        if confidence < confidence_threshold:
            # If confidence is low, ask the user to rephrase
            dispatcher.utter_message("I'm not sure I understood. Could you please rephrase that?")


        return []
    

class ActionProvideCandidateDetailsWithLLM(Action):
    def name(self) -> Text:
        return "action_provide_candidate_profile_details_llm"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]: 

        # Get last message
        last_message = tracker.latest_message.get('text', '')
        
        # get entities
        entities = tracker.latest_message['entities']
        
        candidate_number = None
        # get number of candidate
        for e in entities:
            if e['entity'] == 'candidate_number':
                candidate_number = e['value'].lower()
                break
            
        if candidate_number is None:
            dispatcher.utter_message(text="Please specify the number of the candidate you want to know more about.")
            return []
        else:
        
            # get description of the candidate
            description = DESCRIPTION_USERS[candidate_number]
            
            prompt = PromptTemplate(template=TEMPLATE_RESPONCE_MANAGER, input_variables=["question_manager", "question_manager_template", "context"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            response = llm_chain.run({"question_manager": last_message, "context":description, "question_manager_template": QUESTION_MANAGER_TEMPLATE})
            
            # Send the message back to the user
            dispatcher.utter_message(text=response)
            
            return []