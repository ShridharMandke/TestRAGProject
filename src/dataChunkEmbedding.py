from typing import List, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from src.dataLoader import Load_pdf_documents

class DataEmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1500, chunk_overlap: int = 300):
        """
        Initialize the DataEmbeddingPipeline with a specified embedding model.

        Args:
            model_name (str): The name of the embedding model to use. Default is "all-MiniLM-L6-v2".
        """

        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def chunk_documents(self, documents: List[Any]) -> List[str]:
        """
        Chunk the loaded documents into smaller pieces.

        Args:
            documents (List[Any]): A list of loaded documents.

        Returns:
            List[str]: A list of text chunks.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
            
        #text = doc.page_content  # Assuming each document has a 'page_content' attribute
        dataChunks = splitter.split_documents(documents)
  
        print(f"Documents split into {len(dataChunks)} chunks.")
        
        return dataChunks

    def create_embeddings(self, dataChunks: List[Any]) -> np.ndarray:
        """
        Create embeddings for the given data chunks.

        Args:
            dataChunks (List[Any]): A list of text chunks.

        Returns:
            np.ndarray: An array of embeddings for the text chunks.
        """

        print(f"Creating embeddings for {len(dataChunks)} chunks using model: {self.model.__class__.__name__}")
        data_embeddings = self.model.encode([chunk.page_content for chunk in dataChunks])
        print(f"Created embeddings for {len(data_embeddings)} chunks.")

        return data_embeddings
