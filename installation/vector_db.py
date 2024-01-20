from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class FAISS_db:
    def __init__(self, data_path, loader=CSVLoader):
        self.data_path = data_path
        self.loader = loader
    
    def load_data(self):
        return self.loader(self.data_path).load()
    
    def create_indexes(self, docs):
        return FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    
    def save_db(self, db, path):
        db.save_local(path)

    def load_db(index_path):
        return FAISS.load_local(index_path, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    def search(db, query):
        retriever = db.as_retriever(search_kwargs={"k": 3})
        return retriever.invoke(query)
    

def create_db(data_path, db_path, loader=CSVLoader):
    vdb = FAISS_db(data_path=data_path, loader=loader)
    docs = vdb.load_data()
    db = vdb.create_indexes(docs=docs)
    vdb.save_db(db=db, path=db_path)