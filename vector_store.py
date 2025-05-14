from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

arquivo = "files/apostila.pdf"
loader = PyPDFLoader(arquivo)
paginas = loader.load()

recur_split = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(paginas)
print(len(documents))

embedding_model = OpenAIEmbeddings()

diretorio = "files/chroma_vectorstore"

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=diretorio
)

print(vector_store._collection.count())
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=diretorio
)

pergunta = "Principais métodos para manipulação de strings?"
docs = vector_store.similarity_search(pergunta, k=5)
print(len(docs))

for doc in docs:
    print(doc.page_content)
    print(f"==== {doc.metadata}\n")