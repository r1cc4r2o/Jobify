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
from pathlib import Path

KB_CANDIDATES_PROFILES_PATH = Path("data/KB_user_profiles.csv.gz")


JOBS_VECTOR_DB_PATH = "actions/data/jobs_index"
CANDIDATES_VECTOR_DB_PATH = "actions/data/candidates_index"


# GLOBAL VARIABLES set
DESCRIPTION_USERS = ''

########################################################################################

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

########################################################################################
# Templates LLM
########################################################################################

QUESTION_MANAGER_TEMPLATE = """The manager asks what is in the context above, could you answer his question given the context?"""
TEMPLATE_RESPONCE_MANAGER = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
Answer the question below from context below providing an explanation for the answer:
{context}
{question_manager}
{question_manager_template} [/INST] </s>
"""

TEMPLATE_RESPONCE_MANAGER_profile_summary = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
Summarize the profile of the candidate below with 4 or 5 sentences:
{user_profile}
[/INST] </s>
"""

JOB_TEMPLATE = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
Summarize the profile of the job description below with 2 to 3 sentences:
{job}
[/INST] </s>
"""

########################################################################################
# Load minstral-llm model
########################################################################################

import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, cosine_similarity, sigmoid_kernel, euclidean_distances, manhattan_distances, cosine_distances

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.types import DomainDict
from rasa_sdk.executor import CollectingDispatcher

import pandas as pd

from installation.vector_db import FAISS_db


########################################################################################
########################################################################################

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_4bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config, )
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=600,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline)


########################################################################################

class ActionCustomFallback(Action):
    def name(self) -> Text:
        return "action_custom_fallback"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]: 
        
        dispatcher.utter_message(text="I'm sorry, I didn't understand. Could you please rephrase that?")
        
        return []


########################################################################################


class ValidateLookingForJobForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_looking_for_job_form"

    def validate_job_work_type(
        self, 
        slot_value: Any, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        valid_work_types = ['full_time', 'part_time', 'contract', 'internship', 'temporary']
        if slot_value.lower() in valid_work_types:
            dispatcher.utter_message(text=f"Ok, you are looking for a {slot_value} job.")
            return {"job_work_type": slot_value}
        else:
            dispatcher.utter_message(text="Please specify the type of job you are looking for (full-time, part-time, contract, internship, temporary).")
            return {"job_work_type": None}

    def validate_job_country(
        self, 
        slot_value: Any, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        # Add any specific validation for job country if needed
        return {"job_country": slot_value}

    def validate_job_salary(
        self, 
        slot_value: Any, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        if slot_value.isdigit():
            dispatcher.utter_message(text=f"Ok, you are looking for a job with a salary of {slot_value} €.")
            return {"job_salary": slot_value}
        else:
            dispatcher.utter_message(text="Please specify the salary you are looking for (numeric value).")
            return {"job_salary": None}
        

########################################################################################


class ValidateLookingForCandidateForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_looking_for_candidate_form"
    
    def validate_in_place_or_remote(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict,) -> Dict[Text, Any]:
        if slot_value.lower() in ['remote', 'in_place']:
            # confirm the slot value to the user
            dispatcher.utter_message(text=f"Ok, you are looking for a {slot_value} candidate.")
            return {"in_place_or_remote": slot_value}
        else:
            dispatcher.utter_message(text="Please specify if you are looking for a remote or in place candidate.")
            return {"in_place_or_remote": None}
        
    def validate_candidate_experience(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict,) -> Dict[Text, Any]:
        if slot_value.lower() in ['junior', 'senior', 'expert']:
            # confirm the slot value to the user
            dispatcher.utter_message(text=f"Ok, you are looking for a {slot_value} candidate.")
            return {"candidate_experience": slot_value}
        else:
            dispatcher.utter_message(text="Please specify if you are looking for a junior or senior candidate.")
            return {"candidate_experience": None}

    def validate_candidate_type_position(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict,) -> Dict[Text, Any]:
        if slot_value.lower() in ['full_time', 'part_time', 'internship']:
            # confirm the slot value to the user
            dispatcher.utter_message(text=f"Ok, you are looking for a {slot_value} candidate.")
            return {"candidate_type_position": slot_value}
        else:
            dispatcher.utter_message(text="Please specify if you are looking for a full time or part time candidate.")
            return {"candidate_type_position": None}
        
    def validate_candidate_salary_max(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict,) -> Dict[Text, Any]:
        if slot_value.isdigit():
            # confirm the slot value to the user
            dispatcher.utter_message(text=f"Ok, you are looking for a candidate with a maximum salary of {slot_value} €.")
            return {"candidate_salary_max": slot_value}
        else:
            dispatcher.utter_message(text="Please specify the maximum salary you are willing to pay.")
            return {"candidate_salary_max": None}
        
        

########################################################################################


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
                        f"Name: {db['name'].values[i]} {db['surname'].values[i]} /n" +
                        f"City: {db['city'].values[i]} /n" +
                        f"Relocation: {relocations[i]} /n" +
                        f"Salary: {db['salary'].values[i]} /n" +
                        f"Skills: {db['skills'].values[i]} /n" +
                        f"Mansion: {db['mansion'].values[i]} /n" +
                        f"Part/Full Time: {db['part_full_time'].values[i]} /n" +
                        f"Experience Years: {db['experience_years'].values[i]} /n" + 
                        f"Level Education: {levels_education[i]} /n" +
                        f"Description: {db['description'].values[i]} /n" for i in range(len(db))
                    ] 
                    #    f"Similarity: {db['similarity'].values[i]} /n"
                    
        # join the messages
        last_message = "/n/n".join(last_messages)
        
        # update the description users global variable
        global DESCRIPTION_USERS
        DESCRIPTION_USERS = {i: desc for i, desc in zip(['1', '2', '3', 'one', 'two', 'three'], last_messages + last_messages)}
        
        utter_ask_to_know_more_about_candidate = "Would you like to know more about one of the candidates? Number 1, 2 or 3?"
        last_message = last_message + "/n/n" + utter_ask_to_know_more_about_candidate
        
        # Send the message back to the user
        dispatcher.utter_message(text=last_message)
        
        return []



########################################################################################


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
    


########################################################################################


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
        
            if DESCRIPTION_USERS != '':
                
                # get description of the candidate
                description = DESCRIPTION_USERS[candidate_number]
                
                prompt = PromptTemplate(template=TEMPLATE_RESPONCE_MANAGER, input_variables=["question_manager", "question_manager_template", "context"])
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                response = llm_chain.run({"question_manager": last_message, "context":description, "question_manager_template": QUESTION_MANAGER_TEMPLATE})
                
                # Send the message back to the user
                dispatcher.utter_message(text=response)
                
            else:
                dispatcher.utter_message(text="There is no description of the candidates yet in the memory.")
            
            return []
        
        

######################################################################################## 


class ActionSetJobTitle(Action):
    def name(self) -> Text:
        return "action_set_job_title"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the user input from the latest user message
        user_input = tracker.latest_message.get("text")

        # Set the job_title slot with the extracted job title
        dispatcher.utter_message(text=f"Great! I understand you're looking for '{user_input}' positions.")
        return [SlotSet("job_title", user_input)]
    
    
########################################################################################
  

class ActionSearchJobs(Action):
    def name(self) -> Text:
        return "action_search_jobs"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # requested slots
        job_title = tracker.get_slot("job_title")
        job_work_type = tracker.get_slot("job_work_type")
        job_country = tracker.get_slot("job_country")
        job_salary = tracker.get_slot("job_salary")
        
        MESSAGE_SLOTS = f"""You requested {job_title}, {job_work_type} position with a salary of {job_salary}, in {job_country}."""
        
        query = tracker.latest_message.get('text')
        
        query = MESSAGE_SLOTS + " " + query

        db = FAISS_db.load_db(index_path=JOBS_VECTOR_DB_PATH)
        search_results = FAISS_db.search(db, query)

        if search_results:
    
            message = f"Here are the top job results based on your demand:/n"
            
            for result in search_results:
                
                # get the profile of the candidate
                job = f"{result.page_content}/n"
                
                # create the prompt
                prompt = PromptTemplate(template=JOB_TEMPLATE, input_variables=["job"])
                
                # create the llm chain
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                
                # get the response from the llm model
                response = llm_chain.run({"job": job})
                
                message += f"{response}/n/n"
                
            # Send the message back to the user
            dispatcher.utter_message(text=response)
            
            # update the description users global variable
            global DESCRIPTION_USERS
            DESCRIPTION_USERS = {i: desc for i, desc in zip(['1', '2', '3', 'one', 'two', 'three'], message + message)}
            
            utter_ask_to_know_more_about_job = "Would you like to know more about one of the job positions? Number 1, 2 or 3?"
            message_out = message + "/n/n" + utter_ask_to_know_more_about_job
            
        else:
            message = "I'm sorry, but I couldn't find any matching job positions."

        dispatcher.utter_message(text=message_out)
        
        return []



########################################################################################


class ActionSearchCandidates(Action):
    def name(self) -> Text:
        return "action_search_candidates"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # requested slots
        in_place_or_remote = tracker.get_slot("in_place_or_remote")
        candidate_experience = tracker.get_slot("candidate_experience")
        candidate_type_position = tracker.get_slot("candidate_type_position")
        candidate_salary_max = tracker.get_slot("candidate_salary_max")
        
        MESSAGE_SLOTS = f"""You are looking for a {candidate_experience} candidate for a {candidate_type_position} position with a maximum salary of {candidate_salary_max} USD. The candidate should be {in_place_or_remote}."""
        
        query = tracker.latest_message.get('text')
        
        query = MESSAGE_SLOTS + " " + query

        db = FAISS_db.load_db(index_path=CANDIDATES_VECTOR_DB_PATH)
        search_results = FAISS_db.search(db, query)

        if search_results:
    
            message = f"Here are the top candidates results based on your demand:\n"
            list_candidates = []
            
            for result in search_results:
                
                # get the profile of the candidate
                profile = f"{result.page_content}\n"
                
                # create the prompt
                prompt = PromptTemplate(template=TEMPLATE_RESPONCE_MANAGER_profile_summary, input_variables=["user_profile"])
                
                # create the llm chain
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                
                # get the response from the llm model
                response = llm_chain.run({"user_profile": profile})
                
                message += f"{response}\n\n\n"
                
                list_candidates.append(response)
                
                
            # Send the message back to the user
            dispatcher.utter_message(text=response)
            
            # update the description users global variable
            global DESCRIPTION_USERS
            DESCRIPTION_USERS = {i: desc for i, desc in zip(['1', '2', '3', 'one', 'two', 'three'], list_candidates + list_candidates)}
            
            message_out = message
            
        else:
            message = "I'm sorry, but I couldn't find any matching candidates."

        dispatcher.utter_message(text=message_out)
        
        return []
    
    
########################################################################################


