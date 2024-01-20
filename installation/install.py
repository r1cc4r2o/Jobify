from vector_db import create_db

DATA_PATH = "./data/jobs.csv"
DB_PATH = "../data/jobs_index"

create_db(data_path=DATA_PATH, db_path=DB_PATH)
