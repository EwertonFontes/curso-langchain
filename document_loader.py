from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI


chat = ChatOpenAI(model="gpt-3.5-turbo-0125")


#LENDO ARQUIVOS PDF
arquivo = "files/apostila.pdf"
loader = PyPDFLoader(arquivo)
documentos = loader.load()

len(documentos) #quantidade de paginas


chain = load_qa_chain(llm=chat, chain_type="stuff", verbose=True)
pergunta = "Do que se trata esse documento?"

print(chain.run(input_documents=documentos[:8], question=pergunta))


#LENDO ARQUIVOS CSV
arquivo_csv = "files/imdb_movies.csv"
loader = CSVLoader(arquivo_csv)
documentos = loader.load()

print(documentos[5])

print(len(documentos))

pergunta = "Qual filme com o menos e o maior metascore?"
print(chain.run(input_documents=documentos[:10], question=pergunta))

#CARREGANDO VIDEOS DO YOUTUBE
url = "https://www.youtube.com/watch?v=4p7axLXXBGU"
save_dir = "files/youtube"
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)

docs = loader.load()
print(len(docs))
print(docs[0].page_content[:200])

pergunta = "Faça um resumo brevede sse video para mim"
print(chain.run(input_documents=docs, question=pergunta))

#WEB via URL
web_url = "https://www.alura.com.br/artigos/inteligencia-artificial-ia?utm_term=&utm_campaign=%5BSearch%5D+%5BPerformance%5D+%5BCursos%5D+DSA+-+Forma%C3%A7%C3%B5es&utm_source=google&utm_medium=cpc&campaign_id=21045490451_158851964763_691754664154&utm_id=21045490451_158851964763_691754664154&hsa_acc=7964138385&hsa_cam=%5BSearch%5D+%5BPerformance%5D+%5BCursos%5D+DSA+-+Forma%C3%A7%C3%B5es&hsa_grp=158851964763&hsa_ad=691754664154&hsa_src=g&hsa_tgt=dsa-2276348409543&hsa_kw=&hsa_mt=&hsa_net=google&hsa_ver=3&gad_source=1&gad_campaignid=21045490451&gbraid=0AAAAADpqZIB7Z574mZa618Rt56cgTGzCo&gclid=CjwKCAjw_pDBBhBMEiwAmY02NqYFlVZXKgoQSiWBNo1VsiuV6TDN0J27mgn8xJYYHL2Fkpsp6bdj5xoCcKkQAvD_BwE"
loader = WebBaseLoader(web_url)
documents = loader.load()

print(len(documents))
pergunta = "Faça um resumo breve dessa url"
print(chain.run(input_documents=documents, question=pergunta))