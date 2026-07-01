from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading PDF...")

loader = PyPDFLoader("data/ml1.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

print("Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("Generating embeddings and building FAISS index...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

vectorstore.save_local("vector_store")

print("\n Vector index created successfully!")
print("Saved in: vector_store/")