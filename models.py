from langchain import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

llm = OpenAI(model="gpt-3.5-turbo-instruct")

#TESTE BASICO
prompt = "Conte uma historia sobre aprendizado de maquina"
llm.invoke(prompt)

perguntas = [
    "O Que é memoria RAM?",
    "O que é Disco Rigido?",
    "O que é processador?"
]

llm.batch(perguntas)

#CHAT MODEL
chat = ChatOpenAI(model="gpt-3.5-turbo-0125")
mensagens = [
    SystemMessage(content="Você é um assiste que responde com irônia"),
    HumanMessage(content="Qual o papel da memória cache?")
]

resposta = chat.invoke(mensagens)
print(resposta.content)
print(resposta.response_metadata)

#PROMPT FEW-SHOT
mensagens = [
    HumanMessage(content="Qual é o primeiro dia da semana?"),
    AIMessage(content="Domingo"),
    HumanMessage(content="Qual é o terceiro dia da semana?"),
    AIMessage(content="Terça-Feira"),
    HumanMessage(content="Qual o ultimo dai da semana?")
]
chat.invoke(mensagens)

#CACHEAMENTO
from langchain_openai.chat_models import ChatOpenAI
chat = ChatOpenAI(model="gpt-3.5-turbo-0125")

mensagens = [
    SystemMessage(content="Você é um assistente ironico"),
    HumanMessage(content="Qual é o quinto dia da semana?")
]


set_llm_cache(InMemoryCache())

chat.invoke(mensagens)
chat.invoke(mensagens)

set_llm_cache(SQLiteCache(database_path="files/langchain_cache.sqlite"))

chat.invoke(mensagens)
chat.invoke(mensagens)