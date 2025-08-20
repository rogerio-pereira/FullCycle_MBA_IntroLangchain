from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ```{initial_text}```"
)

template_summarize = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n ```{text}```\n\n"
)

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

translate = template_translate | model | StrOutputParser()
pipeline = translate | (lambda x: {"text": x}) | template_summarize | model | StrOutputParser()

result = pipeline.invoke({"initial_text": "Langchain e um framework para desenvolvimento de aplicações com IA"})

print(result)