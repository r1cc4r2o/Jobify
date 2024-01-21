from langchain.document_loaders import CSVLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

model = 'sentence-transformers/all-MiniLM-L6-v2'

class FAISS_db:
    def __init__(self, data_path):
        self.data_path = data_path
        self.loader = CSVLoader(data_path)
    
    def load_data(self):
        return self.loader.load()
    
    def create_indexes(self, docs):
        print("Creating db...")
        return FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name=model))
    
    def save_db(self, db, path):
        db.save_local(path)

    def load_db(index_path):
        return FAISS.load_local(index_path, HuggingFaceEmbeddings(model_name=model))

    def search(db, query):
        retriever = db.as_retriever(search_kwargs={"k": 3})
        return retriever.invoke(query)
    

def create_db(data_path, db_path):
    vdb = FAISS_db(data_path=data_path)
    docs = vdb.load_data()
    db = vdb.create_indexes(docs=docs)
    vdb.save_db(db=db, path=db_path)