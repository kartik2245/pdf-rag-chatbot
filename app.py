from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context, reply:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()

chain = prompt | llm | parser

print("=" * 60)
print("PDF RAG Chatbot")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        print("\nThank you for using the chatbot!")
        break

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    print("\nRetrieved Chunks")
    print("-" * 60)

    for i, doc in enumerate(results, start=1):
        print(f"\nChunk {i}:")
        print(doc.page_content[:250] + "...")
        print("-" * 60)

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    print("\nAnswer")
    print("-" * 60)
    print(answer)