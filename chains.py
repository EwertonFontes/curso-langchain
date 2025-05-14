from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


chat = ChatOpenAI(model="gpt-3.5-turbo-0125")
memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

chain.predict(input="olá")

prompt_template = PromptTemplate.from_template("""
    Essa é uma conversa amigavel entre um humano e uma IA

    Conversa atual:
    {history}
    Human: {input}
    AI:"""
)

chain = ConversationChain(
    prompt=prompt_template,
    llm=chat,
    memory=memory,
    verbose=True
)

chain.predict(input="Oi")

## LLMCHAIN
prompt = PromptTemplate.from_template(
    """
    Escolha o melhor nome para mim sobre uma empresa que desenvolve soluções em {produto}
"""
)
chain = LLMChain(llm=chat, prompt=prompt)
produto = "LLMs com Langchain"
chain.run(produto)

#SimpleSequentialChain

#SequencialChain
