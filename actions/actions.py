from installation.vector_db import FAISS_db

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

JOBS_VECTOR_DB_PATH = "./data/jobs_index"
CANDIDATES_VECTOR_DB_PATH = "./data/candidates_index"

class ActionSearchJobs(Action):
    def name(self) -> Text:
        return "action_search_jobs"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message.get('text')

        db = FAISS_db.load_db(index_path=JOBS_VECTOR_DB_PATH)
        search_results = FAISS_db.search(db, query)

        if search_results:
            message = f"Here are the top job results based on your query:\n"
            for result in search_results:
                message += f"{result.page_content}\n\n\n"
        else:
            message = "I'm sorry, but I couldn't find any matching jobs."

        dispatcher.utter_message(text=message)
        return []

class ActionSearchCandidates(Action):
    def name(self) -> Text:
        return "action_search_candidates"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message.get('text')

        db = FAISS_db.load_db(index_path=CANDIDATES_VECTOR_DB_PATH)
        search_results = FAISS_db.search(db, query)

        if search_results:
            message = f"Here are the top candidates results based on your query:\n"
            for result in search_results:
                message += f"{result.page_content}\n"
        else:
            message = "I'm sorry, but I couldn't find any matching candidates."

        dispatcher.utter_message(text=message)
        return []
