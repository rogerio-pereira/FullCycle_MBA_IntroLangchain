import os 
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

current_dir = Path(__file__).parent
pdf_path = current_dir / "gpt5.pdf"
C
docs = PyPDFLoader(str(pdf_path)).load()

splits = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=150,
                add_start_index=False
            ).split_documents(docs)

if not splits:
    raise SystemExit(0)

# Runs a loop over all splits, creating a Document at each iteration
#   Each Document will have:
#       page_content from the documents being iterated inside the splits
#       metadata (with the metadata key and value), e.g., title, source, page
#           Only if the value is not empty
enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )

    for d in splits
]

# readable form
#
# enriched = []
# for d in splits:
#     page_content = d.page_content
#
#     meta = {}
#     for k, v in d.metadata.items():
#         if v not in ("", None):
#             meta[k] = v
#
#     doc = Document(
#         page_content=page_content,
#         metadata=meta
#     )
#
#     enriched.append(doc)

ids = [f"doc-{i}" for i in range(len(enriched))]
# Readable form
# ids = []

# for i in range(len(enriched)):
#     id_value = f"doc-{i}"
#     ids.append(id_value)

embeddings = OpenAIEmbeddings(model = os.getenv("OPENAI_MODEL", "text-embedding-3-small"))

store = PGVector.from_documents(
    embeddings = embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection_string=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)
store.add_documents(enriched, ids)