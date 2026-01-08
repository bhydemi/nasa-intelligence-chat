import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [d for d in current_dir.iterdir()
                   if d.is_dir() and (d.name.startswith('chroma') or 'chroma' in d.name.lower())]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}_{collection.name}"

                # Build information dictionary
                # Get document count with fallback for unsupported operations
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = "unknown"

                backends[key] = {
                    # Store directory path as string
                    "directory": str(chroma_dir),
                    # Store collection name
                    "collection_name": collection.name,
                    # Create user-friendly display name
                    "display_name": f"{collection.name} ({doc_count} docs) - {chroma_dir.name}",
                    # Store document count
                    "doc_count": doc_count
                }

        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
            key = f"{chroma_dir.name}_error"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "unknown",
                # Include error information in display name with truncation
                "display_name": f"Error: {str(e)[:50]}... - {chroma_dir.name}",
                "doc_count": 0
            }

    # Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[any, bool, str]:
    """
    Initialize the RAG system with specified backend (cached for performance)

    Args:
        chroma_dir: Path to ChromaDB directory
        collection_name: Name of the collection to use

    Returns:
        Tuple of (collection, success, error_message)
    """
    try:
        # Create a ChromaDB PersistentClient
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Return the collection with the collection_name
        collection = client.get_collection(name=collection_name)
        return collection, True, ""

    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """
    Retrieve relevant documents from ChromaDB with optional filtering

    Args:
        collection: ChromaDB collection object
        query: Search query string
        n_results: Number of results to return
        mission_filter: Optional mission name to filter by (e.g., "apollo_11", "apollo_13", "challenger")

    Returns:
        Query results dictionary or None on error
    """
    try:
        # Initialize filter variable to None (represents no filtering)
        where_filter = None

        # Check if filter parameter exists and is not set to "all" or equivalent
        if mission_filter and mission_filter.lower() not in ["all", "none", ""]:
            # Create filter dictionary with appropriate field-value pairs
            where_filter = {"mission": mission_filter.lower()}

        # Execute database query with the following parameters
        results = collection.query(
            # Pass search query in the required format
            query_texts=[query],
            # Set maximum number of results to return
            n_results=n_results,
            # Apply conditional filter (None for no filtering, dictionary for specific filtering)
            where=where_filter
        )

        # Return query results to caller
        return results

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return None


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """
    Format retrieved documents into context for LLM

    Args:
        documents: List of document text chunks
        metadatas: List of metadata dictionaries for each document

    Returns:
        Formatted context string
    """
    if not documents:
        return ""

    # Initialize list with header text for context section
    context_parts = ["=== RELEVANT NASA MISSION DOCUMENTS ===\n"]

    # Track seen documents to avoid duplicates
    seen_content = set()

    # Loop through paired documents and their metadata using enumeration
    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        # Skip duplicate content
        doc_hash = hash(doc[:200])  # Hash first 200 chars as signature
        if doc_hash in seen_content:
            continue
        seen_content.add(doc_hash)

        # Extract mission information from metadata with fallback value
        mission = metadata.get('mission', 'Unknown Mission')
        # Clean up mission name formatting (replace underscores, capitalize)
        mission_formatted = mission.replace('_', ' ').title()

        # Extract source information from metadata with fallback value
        source = metadata.get('source', 'Unknown Source')

        # Extract category information from metadata with fallback value
        category = metadata.get('document_category', 'General')
        # Clean up category name formatting (replace underscores, capitalize)
        category_formatted = category.replace('_', ' ').title()

        # Create formatted source header with index number and extracted information
        source_header = f"\n--- Source {i + 1}: {mission_formatted} | {category_formatted} | {source} ---\n"
        # Add source header to context parts list
        context_parts.append(source_header)

        # Check document length and truncate if necessary (max 2000 chars per chunk)
        max_length = 2000
        if len(doc) > max_length:
            truncated_doc = doc[:max_length] + "...[truncated]"
            context_parts.append(truncated_doc)
        else:
            # Add full document content to context parts list
            context_parts.append(doc)

    context_parts.append("\n=== END OF DOCUMENTS ===")

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
