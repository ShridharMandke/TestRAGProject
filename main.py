# RAG Pipeline application

from src.dataLoader import Load_pdf_documents
from src.dataChunkEmbedding import DataEmbeddingPipeline
from src.querySearch import QuerySearch
from src.VectorDBStoreEmbeddings import ChromaDBVectorStore


if __name__ == "__main__":

    vectorDBStore = ChromaDBVectorStore(collection_name="PDF_Data")

    if vectorDBStore.collection.count() > 0:
        print("PDF_Data already contains chunks; skipping ingestion.")

    else:
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
        #print("\n\n", dataEmbeddings)

        # Store the embeddings in the vector database
        vectorDBStore = ChromaDBVectorStore()
        vectorDBStore.storeEmbeddingsInVectorDB(pdf_documents, dataChunks, dataEmbeddings)

    querySearch = QuerySearch()
    userQuery = "What are JLR February 2018 wholesale numbers?"
    
    search_results = querySearch.searchVectorDBUsingQuery(userQuery)
    augmented_query = querySearch.augmentUserQueryWithSearchResults(userQuery, search_results)

    querySearch.getQueryResponsefromLLM(userQuery, augmented_query)
