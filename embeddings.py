from langchain_openai import OpenAIEmbeddings
import numpy as np

embedding_model = OpenAIEmbeddings()
embeddings = embedding_model.embed_documents(
    [
        "Eu gosto de cachorro",
        "Eu gosto de animais",
        "Hoje a tarde está chovendo"
    ]
)

print(len(embeddings))

print(np.dot(embeddings[0], embeddings[1]))
print(np.dot(embeddings[0], embeddings[2]))
print(np.dot(embeddings[1], embeddings[2]))

for i in range(len(embeddings)):
    for j in range(len(embeddings)):
        print(round(np.dot(embeddings[i], embeddings[j]), 2), end=" | ")

pergunta = "O que é um cachorro?"
emb_query = embedding_model.embed_query(pergunta)
emb_query[:10]