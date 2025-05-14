from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain

memory = ConversationBufferMemory(return_messages=True)

memory.chat_memory.add_user_message("Olá")
memory.chat_memory.add_ai_message("Como vai?")

memory.load_memory_variables({})

chat = ChatOpenAI()
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

conversation.predict("Olá, meu nome é Ewerton")
conversation.predict("Como vai?")
conversation.predict("Como é meu nome?")