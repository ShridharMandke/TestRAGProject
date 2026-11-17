import os
from typing import List, Any
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from src.dataChunkEmbedding import DataEmbeddingPipeline

class ChromaDBVectorStore:
    def __init__(self, collection_name: str = "PDF Data", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1500, chunk_overlap: int = 300):
        """
        Initialize the ChromaDBVectorStore with a specified collection name.

        Args:
            collection_name (str): The name of the ChromaDB collection to use. Default is "my_collection".
        """

        self.client = chromadb.PersistentClient(path="G:\\Shridhar Projects\\Test PDFs\\ChromaDB")
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"Initialized ChromaDBVectorStore with collection: {self.collection_name} and embedding model: {self.embedding_model}")

    def storeEmbeddingsInVectorDB(self, documents: list[Any], dataChunks: list[str], dataEmbeddings: np.ndarray):
        """
        Add document embeddings to the ChromaDB collection.

        Args:
            documents: A list of loaded documents.
            dataChunks: A list of text chunks corresponding to the documents.
            dataEmbeddings: A list of document embeddings to add to the collection.
        """

        # Add the embeddings and corresponding metadata to the ChromaDB collection
        self.collection.add(
            documents=[chunk.page_content for chunk in dataChunks],
            embeddings=dataEmbeddings.tolist(),
            metadatas=[{"source": chunk.source for chunk in dataChunks}, {"page_number": chunk.metadata.get("page_label", None) for chunk in dataChunks}]
        )