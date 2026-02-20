import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
DB_FAISS_PATH = os.path.join("vectorstore", "db_faiss")

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"⚠️  Created folder: '{DATA_PATH}'.")
        print(f"Action: Please paste your medical PDF inside the 'data' folder on your Desktop.")
        return

    print("--- [1/4] Loading PDF Files... ---")
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print(f"❌ No PDF found in '{DATA_PATH}'. Please add a file.")
        return

    print(f"--- [2/4] Splitting {len(documents)} Documents... ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("--- [3/4] Creating Embeddings... ---")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    print("--- [4/4] Saving Vector DB... ---")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Success! Database ready.")

if __name__ == "__main__":
    create_vector_db()