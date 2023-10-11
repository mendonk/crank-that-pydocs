import os
import uuid
import cassio
from typing import List
from bs4 import BeautifulSoup
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

def getCQLSession():
    cluster = Cluster(
        cloud={
            "secure_connect_bundle": os.environ['ASTRA_DB_SECURE_BUNDLE_PATH'],
        },
        auth_provider=PlainTextAuthProvider(
            "token",
            os.environ['ASTRA_DB_APPLICATION_TOKEN'],
        ),
    )
    return cluster.connect()

# Initialize cassIO
cassio.init(
    token=os.environ['ASTRA_DB_APPLICATION_TOKEN'],
    database_id=os.environ['ASTRA_DB_ID'],
    keyspace=os.environ['ASTRA_DB_KEYSPACE']
)

# Scrape the listed website
loader = WebBaseLoader("https://docs.langstream.ai")
loader.requests_kwargs = {'verify': False}
documents = loader.load()

# Chunk text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# Compute embeddings with Hugging Face
class EmbeddingsGenerator:
    def __init__(self, model_name: str, model_kwargs=None):
        self.client = SentenceTransformer(model_name, **model_kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ensure all texts are strings
        texts = [str(text).replace("\n", " ") for text in texts]
        return self.client.encode(texts).tolist()

embeddings_generator = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
embeddings = embeddings_generator.embed_documents(texts)

# Output to AstraDB vector database
astraSession = getCQLSession()
astraSession.set_keyspace(os.environ['ASTRA_DB_KEYSPACE'])

for text, embedding in zip(texts, embeddings):
    doc_id = uuid.uuid4()  # Generate a unique ID
    actual_text_list = [text.page_content]
    print(type(text), text)
    print(type(embedding), embedding)
    astraSession.execute(
        """
        INSERT INTO tbl2 (doc_id, list_text, embedding)
        VALUES (%s, %s, %s)
        """,
        (doc_id, actual_text_list, embedding)
    )
    print("*", end='')
