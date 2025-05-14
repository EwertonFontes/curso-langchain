from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

arquivos = [
    "files/apostila.pdf",
    "files/LLM.pdf"
    ]

paginas = []

for arquivo in arquivos:
    loader = PyPDFLoader(arquivo)
    paginas.extend(loader.load())

recur_split = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(paginas)


for i, doc in enumerate(documents):
    doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '')
    doc.metadata['doc_id'] = i

embeddings_model = OpenAIEmbeddings()

diretorio = 'files/chroma_retrival_bd'

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=diretorio
)

# SEMANTIC SEARCH
pergunta = "O que é LLM?"
docs = vectordb.similarity_search(pergunta, k=3)
for doc in docs:
    print(doc.page_content)
    print(f"========{doc.metadata}\n")

#MAX Margina Relevance
pergunta = "O que é LLM?"
docs = vectordb.max_marginal_relevance_search(pergunta, k=3, fetch_k=10)
for doc in docs:
    print(doc.page_content)
    print(f"========{doc.metadata}\n")

#FILTRAGEM
pergunta = "O que a apostila de LLM fala sobre a OpenAI e o ChatGPT?"

docs = vectordb.similarity_search(
    pergunta,
    k=3,
    filter={"source": "LLM.pdf"}
)
for doc in docs:
    print(doc.page_content)
    print(f"========{doc.metadata}\n")


pergunta = 'O que a apostila de LLM fala sobre a OpenAI e o ChatGPT?'

docs = vectordb.similarity_search(
    pergunta, 
    k=3,
    filter={'$and':
            [{'source': {'$in': ['LLM.pdf']}},
            {'page': {'$in': [3, 4, 5, 6]}}],
            })
for doc in docs:
    print(doc.page_content)
    print(f'==========={doc.metadata}\n\n')

#LLM AIDED RETRIEVEL
metadata_info = [
    AttributeInfo(
        name='source',
        description='Nome da apostila de onde o texto original foi retirado. Pode ser "apostila.pdf" ou "LLM.pdf".',
        type='string'
    ),
    AttributeInfo(
        name='page',
        description='A página da apostila de onde o texto foi extraído. Número da página.',
        type='integer'
    ),
]

document_description = 'Apostilas de informações'
llm = OpenAI()
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_description,
    metadata_info,
    verbose=True
)

pergunta = 'O que a apostila de LLM fala sobre a OpenAI, ChatGPT e Hugging Face?'

docs = retriever.get_relevant_documents(pergunta)
for doc in docs:
    print(doc.page_content)
    print(f'==========={doc.metadata}\n\n')