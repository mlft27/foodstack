"""RAG (Retrieval-Augmented Generation) tools for menu retrieval."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from foodstack.data.menu import menu_catalog

def create_menu_documents() -> list[Document]:
    """Convert menu items to LangChain Documents.

    Returns:
        List of Document objects containing menu item information.
    """
    documents = []
    for item in menu_catalog:
        content = f"""
            Name: {item['name']}
            Category: {item['category']}
            Price: ${item['price']}
            Rating: {item['rating']}/5
            Dietary Tags: {item['dietary_tags'] if item['dietary_tags'] else 'None'}
            Description: {item['description']}
            Available: {'Yes' if item['available'] else 'No'}
            Item ID: {item['id']}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "price": item["price"],
            },
        )
        documents.append(doc)

    return documents


def create_chroma_vector_store(
    documents: list[Document] | None = None,
    persist_directory: str = "./chroma_db",
) -> Chroma:
    """Create and return a ChromaDB vector store for menu items.

    Args:
        documents: List of documents to add. If None, uses default menu items.
        persist_directory: Directory to persist the vector store.

    Returns:
        Chroma vector store instance.
    """
    if documents is None:
        documents = create_menu_documents()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="foodstack_menu",
    )

    return vector_store


def get_menu_retriever(
    documents: list[Document] | None = None,
    persist_directory: str = "./chroma_db",
    k: int = 3,
):
    """Get a retriever for menu items.

    Args:
        documents: List of documents to add. If None, uses default menu items.
        persist_directory: Directory to persist the vector store.
        k: Number of results to return per query.

    Returns:
        A retriever that can search menu items by similarity.
    """
    vector_store = create_chroma_vector_store(documents, persist_directory)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever


# Initialize default retriever
def initialize_menu_retriever() -> None:
    """Initialize the default menu retriever for use in the application."""
    global menu_retriever
    menu_retriever = get_menu_retriever()


menu_retriever = None
