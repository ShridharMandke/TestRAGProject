# RAG Pipeline application

from src.dataLoader import Load_pdf_documents
from src.dataChunkEmbedding import DataEmbeddingPipeline


if __name__ == "__main__":

    pdf_directory = "G:\\Shridhar Projects\\Test PDFs"
    pdf_documents = Load_pdf_documents(pdf_directory)
    #print(pdf_documents)

    for doc in pdf_documents:
        print("Metadata:", doc.metadata)
        print("Text attributes:", doc.page_content[:200])

    # Initialize the DataEmbeddingPipeline
    embeddingPipeline = DataEmbeddingPipeline()

    # Prepare Data Chunks
    dataChunks = embeddingPipeline.chunk_documents(pdf_documents)
    #print("\n\n", dataChunks)

    # Create embeddings for the data chunks
    dataEmbeddings = embeddingPipeline.create_embeddings(dataChunks)
    print("\n\n", dataEmbeddings)


