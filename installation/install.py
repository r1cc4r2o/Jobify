import os

from vector_db import create_db

DATA_PATH = "./installation/data/jobs.csv"
DB_PATH = "./data/jobs_index"

if not os.path.exists("../data/"):
    os.mkdir("../data/")

create_db(data_path=DATA_PATH, db_path=DB_PATH)
