import os
from google import genai
from typing import List, Any
from src.VectorDBStoreEmbeddings import ChromaDBVectorStore
from src.dataChunkEmbedding import DataEmbeddingPipeline
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class QuerySearch:
    
    def __init__(self, collection_name: str = "PDF_Data", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the QuerySearch with a specified collection name and embedding model.

        Args:
            collection_name (str): The name of the ChromaDB collection to use. Default is "PDF_Data".
            embedding_model (str): The name of the embedding model to use. Default is "all-MiniLM-L6-v2".
        """

        self.vectorDBStore = ChromaDBVectorStore(collection_name=collection_name, embedding_model=embedding_model)
        self.embedding_model = SentenceTransformer(embedding_model)

    def searchVectorDBUsingQuery(self, query: str, top_k: int = 5) -> list[Any]:
        """
        Search the vector database using the provided query and retrieve relevant documents from the ChromaDB collection.

        Args:
            query (str): The user query to search for relevant documents.
            top_k (int): The number of top relevant documents to retrieve. Default is 5.
        """
        
        #Create embedding for the query
        query_embedding = self.embedding_model.encode([query])

        # Retrieve relevant documents from the ChromaDB collection
        results = self.vectorDBStore.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )

        print (f"Retrieved {len(results)} relevant documents for the query: '{query}'")

        search_results = []
        for i, (doc_id, distance, content, metadata) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['documents'][0],
            results['metadatas'][0]
        )):
            similarity_score = 1 - distance  # Convert distance to similarity score
            search_results.append({
                "doc_id": doc_id,
                "similarity_score": similarity_score,
                "content": content,
                "metadata": metadata
            })

            print(f"\n{i+1}. {metadata.get('source', 'Unknown Source')}")
            print(f"\n   Similarity Score: {similarity_score:.3f}")
            print(f"\n   Content: {content[:200]}...")  # Print first 200 characters of content

        return search_results

    def augmentUserQueryWithSearchResults(self, user_query: str, search_results: List[Any], top_k: int = 5) -> str:
        """
        Augment the user query with relevant search results from the ChromaDB collection.

        Args:
            user_query (str): The user query to augment.
            top_k (int): The number of top relevant documents to retrieve. Default is 5.
        
        """

        # Sort the search results based on similarity score in descending order
        sorted_results = sorted(search_results, key=lambda x: x['similarity_score'], reverse=True)

        # Select the top_k relevant documents
        top_results = sorted_results[:top_k]

        context_parts = []
        for i, result in enumerate(top_results):
            context_parts.append(f"Source {i+1}: {result['metadata'].get('source', 'Unknown Source')}\n {result['content'][:200]}...")  # Print first 200 characters of content

        context = "\n\n".join(context_parts)

        # Concatenate the content of the top results to augment the user query
        augmented_query = f"""
            Based on following context, answer the user's query:

            CONTEXT: {context}

            USER QUERY: {user_query}
        """
        # for i, result in enumerate(top_results):
        #     augmented_query += f"\n{i+1}. Source: {result['metadata'].get('source', 'Unknown Source')}\n"
        #     augmented_query += f"   Similarity Score: {result['similarity_score']:.3f}\n"
        #     augmented_query += f"   Content: {result['content'][:200]}...\n"  # Print first 200 characters of content

        print(f"\nAugmented Query:\n{augmented_query}")

        return augmented_query

    def getQueryResponsefromLLM(self, user_query: str, augmented_query: str) -> str:
        """
        Get a response from the LLM based on the augmented query.

        Args:
            user_query (str): The original user query.
            augmented_query (str): The augmented query to send to the LLM.
        """
        
        # Placeholder for LLM response generation
        # In a real implementation, you would call the LLM API here and return the response.
        # For example:
        # response = llm_api.generate_response(augmented_query)
        # return response

        #api_key = "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"

        self.client = genai.Client(api_key="AQ.Ab8RN6LYF_RH9axyrzY8l8g_4fMCcBrwMHtz4hRX5ysvtfuLZA")

        print(f"\nSending augmented query to LLM:\n{augmented_query}")

        print(f"\n\nBased on user query: '{user_query}', below are the details found in the PDF documents:\n")

        interaction = self.client.interactions.create(
            model="gemini-4-hydrogen",
            input=augmented_query
        )

        #response = self.llm.invoke(augmented_query)

        print(f"\nResponse:\n{interaction.output_text}")