from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
llm = OpenAI()

prompt_template = PromptTemplate.from_template("""
    Responda a seguinte pergunta do usuário em até {n_palavras} palavras:
    {pergunta}
""", partial_variables={"n_palavras": 10})

print(prompt_template.format(pergunta="O que é um SaaS?", n_palavras=15))


# UTILIZANDO MULTIPLOS PROMPTS
template_word_count = PromptTemplate.from_template("""
Responda a pergunta em até {n_palavras} palavras.
"""
)

template_line_count = PromptTemplate.from_template("""
Responda a pergunta em até {n_linhas} linhas.
""")

template_idioma = PromptTemplate.from_template("""
Retorne a resposta em  {idioma} idioma.
""")


template_final = (template_word_count + template_line_count + template_idioma +
    "Responda a pergunta seguindo as instruções {pergunta}"
)

prompt_final = template_final.format(n_palavras=15, idiota="Inglês", n_linhas=3, pergunta="O que é LangChain")
llm.invoke(prompt_final)

#TEMPLATES PARA CHAT
chat_template = ChatPromptTemplate.from_template("Essa é minha dúvida: {duvida}")
chat_template.format_messages(duvida="Quem é você?")

chat_template = ChatPromptTemplate.from_template([
    ("system", "Você é um assistente irônico e se chama {nome_assistente}"),
    ("humam", "Olá, como vai?")
    ("ai", "Estou bem, como posso lhe ajudar?")
    ("humam", "{pergunta}")
])

chat_template.format_messages(nome_assistente="Bot", pergunta="Qual seu nome?")

chat = ChatOpenAI()
chat.invoke(chat_template.format_messages(nome_assistente="Bot", pergunta="Qual seu nome?"))

#FEW-Shot Prompt
exemplos = [
    {"pergunta": "Qual é a maior montanha do mundo, o Monte Everest ou o K2?", 
     "resposta": 
     """São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Qual é a altura do Monte Everest? 
Resposta intermediária: O Monte Everest tem 8.848 metros de altura. 
Pergunta de acompanhamento: Qual é a altura do K2? 
Resposta intermediária: O K2 tem 8.611 metros de altura. 
Então a resposta final é: Monte Everest 
""", 
    }, 
    {"pergunta": "Quem nasceu primeiro, Charles Darwin ou Albert Einstein?", 
     "resposta": 
     """São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Quando nasceu Charles Darwin? 
Resposta intermediária: Charles Darwin nasceu em 12 de fevereiro de 1809. 
Pergunta de acompanhamento: Quando nasceu Albert Einstein? 
Resposta intermediária: Albert Einstein nasceu em 14 de março de 1879. 
Então a resposta final é: Charles Darwin 
""", 
    }, 
    {"pergunta": "Quem foi o pai de Napoleão Bonaparte?",
     "resposta": 
     """São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Quem foi Napoleão Bonaparte? 
Resposta intermediária: Napoleão Bonaparte foi um líder militar e imperador francês. 
Pergunta de acompanhamento: Quem foi o pai de Napoleão Bonaparte? 
Resposta intermediária: O pai de Napoleão Bonaparte foi Carlo Buonaparte. 
Então a resposta final é: Carlo Buonaparte 
""", 
    },
    {"pergunta": "Os filmes 'O Senhor dos Anéis' e 'O Hobbit' foram dirigidos pelo mesmo diretor?", 
     "resposta": 
     """São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Quem dirigiu 'O Senhor dos Anéis'? 
Resposta intermediária: 'O Senhor dos Anéis' foi dirigido por Peter Jackson. 
Pergunta de acompanhamento: Quem dirigiu 'O Hobbit'? 
Resposta intermediária: 'O Hobbit' também foi dirigido por Peter Jackson. 
Então a resposta final é: Sim 
""",
    },
]

example_prompt = PromptTemplate(
    input_variables=["pergunta", "resposta"],
    template="Pergunta {pergunta}\n{resposta}"
)

example_prompt.format(**exemplos[0])

prompt = FewShotPromptTemplate(
    examples=exemplos,
    example_prompt=example_prompt,
    suffix="Pergunta: {input}",
    input_variables=["input"]
)

print(prompt.format(input="Quem é melhor, Messi ou Cristiano Ronaldo?"))
llm.invoke(prompt.format(input="Quem fez mais gols, Messi ou Cristiano Ronaldo?"))

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{pergunta}"),
     ("ai", "{resposta}")]
)

print(example_prompt.format_messages(**exemplos[0]))

few_shot_template = FewShotChatMessagePromptTemplate(
    examples=exemplos,
    example_prompt=example_prompt
)

prompt_final = ChatPromptTemplate.from_messages([
    few_shot_template,
    ("human", "{input}")
])

prompt = prompt_final.format_messages(input="Quem fez mais gols, Messi ou Cristiano Ronaldo?")
# few_shot_template.format_messages()
chat.invoke(prompt)