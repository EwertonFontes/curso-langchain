from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

feedback_produto = """
Estou muito satisfeito com o Smartphone XYZ Pro. O desempenho é excelente, e o sistema 
operacional é rápido e intuitivo. A câmera é um dos principais destaques, especialmente o 
modo noturno, que captura imagens incríveis mesmo em baixa iluminação. A duração da bateria 
também impressiona, durando facilmente um dia inteiro com uso intenso.
Por outro lado, sinto que o produto poderia ser melhorado em alguns aspectos. A tela, 
embora tenha cores vibrantes, parece refletir bastante luz, dificultando o uso sob o sol. 
Além disso, o carregador incluído na caixa não oferece carregamento rápido, o que é um ponto 
negativo considerando o preço do aparelho
"""


review_template = ChatPromptTemplate.from_template("""
Para o texto a seguir, extraia as seguintes informações:
produto: Nome do produto mencionado no texto.
características_positivas: Liste todas as características positivas mencionadas sobre o produto.
características_negativas: Liste todas as características negativas mencionadas sobre o produto.
recomendação: O cliente recomenda o produto? Responda True para sim ou False para não.

Texto: {review}

Retorne a resposta no formato JSON
"""
)

print(review_template.format_messages(review=feedback_produto))

chat = ChatOpenAI()
resposta = chat.invoke(review_template.format_messages(review=feedback_produto))
resposta.content

schema_produto = ResponseSchema(
    name="produto",
    type="string",
    description="Nome do produto mencionado no texto"
)


schema_positivas = ResponseSchema(
    name="caracteristicas_positivas",
    type="list",
    description="Liste todas as características positivas mencionadas sobre o produto"
)

schema_negativas = ResponseSchema(
    name='características_negativas',
    type='list',
    description='Liste todas as características negativas mencionadas sobre o produto.'
)

schema_recomendacao = ResponseSchema(
    name='recomendação',
    type='bool',
    description='O cliente recomenda o produto? Responda True para sim ou False para não.'
)

response_schema = [schema_produto, schema_positivas, schema_negativas, schema_recomendacao]
output_parser = StructuredOutputParser.from_response_schemas(response_schema)
schema_formatado = output_parser.get_format_instructions()
print(schema_formatado)

review_template2 = ChatPromptTemplate.from_template("""
Para o texto a seguir, extraia as seguintes informações:
produto, caracteristicas_positivas, caracteristicas_negativas e recomendacao

Texto: {review}

{schema}
""", partial_variables={"schema":schema_formatado}
)

print(review_template2.format_messages(review=feedback_produto))

resposta = chat.invoke(review_template2.format_messages(review=feedback_produto))
print(resposta.content)

resposta_json = output_parser.parse(resposta.content)
print(resposta_json)