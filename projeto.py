from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug

chat = ChatOpenAI(model="gpt-4o-mini")

caminhos = [
    "files/apostila.pdf",
    "files/LLM.pdf",
    ]

paginas = []
for caminho in caminhos:
    loader = PyPDFLoader(caminho)
    paginas.extend(loader.load())

recur_split = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(paginas)

for i, doc in enumerate(documents):
    doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '')
    doc.metadata['doc_id'] = i

diretorio = 'arquivos/chat_retrieval_db'

embeddings_model = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=diretorio
)

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectordb.as_retriever(search_type='mmr'),
)

pergunta = "O que é Hugging Face e como faço para acessá-lo?"
chat_chain.invoke({"query": pergunta})

chain_prompt = PromptTemplate.from_template(
"""Utilize o contexto fornecido para responder a pergunta ao final. 
Se você não sabe a resposta, apenas diga que não sabe e não invente uma resposta.
Utilize três frases no máximo, mantenha a resposta concisa.

Contexto: {context}

Pergunta: {question}

Resposta:
"""
)

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectordb.as_retriever(search_type="mmr"),
    chain_type_kwargs={"prompt":chain_prompt},
    return_source_documents=True
)

pergunta = 'O que é Hugging Face e como faço para acessá-lo?'
resposta = chat_chain.invoke({'query': pergunta})
print(resposta['result'])

set_debug(True)

pergunta = 'O que é Hugging Face e como faço para acessá-lo?'
resposta = chat_chain.invoke({'query': pergunta})

set_debug(False)

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectordb.as_retriever(search_type='mmr'),
    chain_type='refine'
)

pergunta = 'O que é Hugging Face e como faço para acessá-lo?'
resposta = chat_chain.invoke({'query': pergunta})
print(resposta['result'])