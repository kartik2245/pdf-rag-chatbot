from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("pdf/ml1.pdf")
documents = loader.load()
print(len(documents))
print(documents[0])