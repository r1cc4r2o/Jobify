## Greet and Ask for Purpose
* greet
  - utter_greet
* choose_purpose
  - utter_ask_purpose

## User is Looking for a Job
* user_wants_job
  - utter_ask_job_search_conditions

## Provide Job Search Conditions
* provide_job_search_conditions{"experience": "2-4 years", "location": "New York"}
  - action_search_jobs
  - utter_job_results
  - checkpoint_after_job_search
  - utter_ask_another_search

## User Wants to Look for Another Job
* user_wants_another_job
  - checkpoint_after_job_search
  - utter_ask_job_search_conditions

## User is Looking for a Candidate
* user_wants_candidate
  - utter_ask_candidate_search_conditions

## Provide Candidate Search Conditions
* provide_candidate_search_conditions{"experience": "5+ years", "skills": "python"}
  - action_search_candidates
  - utter_candidate_results
  - checkpoint_after_candidate_search
  - utter_ask_another_search

## User Wants to Look for Another Candidate
* user_wants_another_candidate
  - checkpoint_after_candidate_search
  - utter_ask_candidate_search_conditions

## Continue Searching
* continue_search{"continue": "yes"}
  - utter_ask_purpose

## Stop Searching
* continue_search{"continue": "no"}
  - utter_goodbye
