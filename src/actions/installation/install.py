import os

from vector_db import create_db

JOBS_DATA_PATH = "./src/installation/data/jobs.csv"
JOBS_DB_PATH = "./src/data/jobs_index"
CANDIDATES_DATA_PATH = "./installation/data/candidates.csv"
CANDIDATES_DB_PATH = "./data/candidates_index"

if not os.path.exists("../data/"):
    os.mkdir("../data/")

create_db(data_path=JOBS_DATA_PATH, db_path=JOBS_DB_PATH)
create_db(data_path=CANDIDATES_DATA_PATH, db_path=CANDIDATES_DB_PATH)
