import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissChatbot:
    def __init__(self, index_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(embedding_model)

    def query(self, question):
        vector = self.model.encode([question])[0]
        D, I = self.index.search(np.array([vector]).astype('float32'), k=1)
        return D[0][0], I[0][0]

def main():
    index_path = "faiss/index.faiss"
    chatbot = FaissChatbot(index_path)

    while True:
        try:
            question = input("Ask a question: ")
            distance, index = chatbot.query(question)
            # For now, we're returning the index of the most similar information. 
            # In a real-world scenario, you might want to retrieve the actual information using this index.
            print(f"Your answer can be found in document ID: {index}")
        
        except KeyboardInterrupt:
            print("\nExiting the chatbot.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
