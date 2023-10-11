from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

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

# Create FAISS index for embeddings
dimension = len(embeddings[0])  # Assuming all embeddings have the same size
index = faiss.IndexFlatL2(dimension)
embeddings_array = np.array(embeddings)
index.add(embeddings_array)

# Map from FAISS's integer index to actual text
index_to_text = {i: text for i, text in enumerate(texts)}

# Save the FAISS index locally
faiss.write_index(index, "faiss_index.index")
print("FAISS index and mapping saved.")
