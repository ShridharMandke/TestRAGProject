from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from typing import List, Any
from pathlib import Path


def Load_pdf_documents(pdf_directory: str) -> List[Any]:
    """
    Load PDF documents from a specified directory.

    Args:
        pdf_directory (str): The path to the directory containing PDF files.    

    Returns:
        List[Any]: A list of loaded PDF documents.
    """

    documents = []

    data_path = Path(pdf_directory).resolve()
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in the directory: {pdf_directory}")

    for pdf_file in pdf_files:
        print(f"Loading PDF file: {pdf_file}")

        try:
            loader = PyPDFLoader(pdf_file)
            loaded_documents = loader.load()
            documents.extend(loaded_documents)
            print(f"Successfully loaded {len(loaded_documents)} documents from {pdf_file}")

        except Exception as e:
            print(f"Error loading PDF file {pdf_file}: {e}")

    return documents