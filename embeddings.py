from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text1 = "Artificial Intelligence"

text2 = "AI"

text3 = "Pizza is delicious"
vector1 = embeddings.embed_query(text1)
vector2 = embeddings.embed_query(text2)
vector3 = embeddings.embed_query(text3)

similarity1 = cosine_similarity([vector1], [vector2])

similarity2 = cosine_similarity([vector1], [vector3])

print(similarity1)
print(similarity2)

print(type(vector1))
print(len(vector1))
print(vector1[:10])
print(type(vector2))
print(len(vector2))
print(vector2[:10])
print(type(vector3))
print(len(vector3))
print(vector3[:10])