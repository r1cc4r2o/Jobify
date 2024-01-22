## User Wants a Job
- intent: user_wants_job
  - action: utter_ask_job_search_conditions

## Provide Job Search Conditions
- intent: provide_job_search_conditions
  - action: action_search_jobs
  - action: utter_job_results
  - checkpoint: checkpoint_after_job_search
  - action: utter_ask_another_search

## User Wants Another Job
- intent: user_wants_another_job
  - checkpoint: checkpoint_after_job_search
  - action: utter_ask_job_search_conditions

## User Wants a Candidate
- intent: user_wants_candidate
  - action: utter_ask_candidate_search_conditions

## Provide Candidate Search Conditions
- intent: provide_candidate_search_conditions
  - action: action_search_candidates
  - action: utter_candidate_results
  - checkpoint: checkpoint_after_candidate_search
  - action: utter_ask_another_search

## User Wants Another Candidate
- intent: user_wants_another_candidate
  - checkpoint: checkpoint_after_candidate_search
  - action: utter_ask_candidate_search_conditions

## Continue Searching
- intent: continue_search
  continue: "yes"
  - action: utter_ask_purpose

## Stop Searching
- intent: continue_search
  continue: "no"
  - action: utter_goodbye
