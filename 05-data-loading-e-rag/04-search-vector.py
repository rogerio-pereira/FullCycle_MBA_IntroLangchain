import os 
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()
for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

query = "Tell me more about gpt-5 thinking evaluation and performance results comparing to gpt-4"

gemini_model = os.getenv("GOOGLE_MODEL", "models/embedding-001")
embeddings = GoogleGenerativeAIEmbeddings(
    model=gemini_model,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

# k is the quantity of documents
results = store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, start=1):
    print("="*50)
    print(f"Result {i} (score: {score:.2f}):")
    print("="*50)

    print(f"\nText:\n")
    print(doc.page_content.strip())

    print(f"\nMetadata:\n")
    for key, value in doc.metadata.items():
        print(f"{key}: {value}")