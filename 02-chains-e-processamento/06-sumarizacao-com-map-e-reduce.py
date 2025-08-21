from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

long_text = """
Artificial Intelligence (AI) represents one of the most significant technological developments of our time. The concept was first introduced at the 1956 Dartmouth Conference, where scientists gathered to explore the possibility of creating machines that could simulate human thinking. Early AI systems relied on symbolic logic and rule-based approaches, but these proved limited when dealing with real-world complexity.

The modern AI revolution began in the 1990s with machine learning, which allows computers to learn patterns from data rather than following explicit programming rules. This approach led to breakthroughs in speech recognition, computer vision, and natural language processing. Deep learning, a subset of machine learning using neural networks, emerged as a powerful tool in the 2010s, enabling significant advances in image and audio analysis.

Today, AI applications are widespread across industries. Healthcare systems use AI to analyze medical images and detect diseases. Financial institutions employ AI for fraud detection and risk assessment. Transportation companies are developing self-driving vehicles, while retail businesses use AI for customer service and product recommendations.

However, AI development also raises important concerns about job displacement, algorithmic bias, and privacy. The future of AI depends on addressing these challenges while continuing to advance the technology's capabilities. Success will require developing AI that augments human abilities rather than replacing them entirely, ensuring that the technology benefits society as a whole.
"""


# RecursiveCharacterTextSplitter e mais inteligente, ele entende e prioriza quebra de linha, quebra de linha duplas, 
#       sentencas, e busca paragrafos e pedacos para fazer o splitting do texto. 
#       ele evita quebrar palavras no meio, e tenta manter o texto inteiro. geralmente parando em espacos e pontuacoes
#           pode acontecer de quebrar palavras muito grandes no meio (por exemplo, URLs)
# CharacterTextSplitter e mais simples e geralmente corta palavras no meio
#       Pode passar regras para ele decidir onde cortar o texto, por exemplo \n (porem isso pode cortar de forma nao
#            equilibrada)
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=70)

parts = splitter.create_documents([long_text])

# for part in parts:
#     print(part.page_content)
#     print("-"*30)

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

chain_summarize = load_summarize_chain(model, chain_type="map_reduce", verbose=False)

result = chain_summarize.invoke({"input_documents": parts})

print(result["output_text"])