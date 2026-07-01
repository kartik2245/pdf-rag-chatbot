from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader("pdf/ml1.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print(f"Number of Chunks: {len(chunks)}")
print(chunks[0])
print(chunks[0].page_content)
print(documents[0].page_content)