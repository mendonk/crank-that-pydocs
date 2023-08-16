import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissVectorQuery:
    def __init__(self, index_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(embedding_model)

    def search(self, query_text, k=5):
        vector = self.model.encode([query_text])[0]
        distances, indices = self.index.search(np.array([vector]).astype('float32'), k)
        return distances[0], indices[0]

def main():
    index_path = "faiss/index.faiss"
    vector_query = FaissVectorQuery(index_path)

    while True:
        try:
            query = input("\nEnter your query text (or type 'exit' to quit): ")
            
            if query.lower() == 'exit':
                break
            
            distances, indices = vector_query.search(query)
            print(f"Top {len(indices)} matching vector indices: {indices}")
            print(f"Distances for these vectors: {distances}")
        
        except KeyboardInterrupt:
            print("\nExiting the program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
