# NASA Intelligence Chat System

A complete Retrieval-Augmented Generation (RAG) system for querying NASA space mission documents, including Apollo 11, Apollo 13, and Space Shuttle Challenger missions.

## Overview

This system allows users to ask questions in plain English about NASA's historic space missions and receive accurate, detailed answers sourced directly from NASA's archives. The system uses:

- **ChromaDB** for vector storage and semantic search
- **OpenAI** for embeddings and language model responses
- **RAGAS** for real-time response quality evaluation
- **Streamlit** for an interactive chat interface

## Project Structure

```
nasa-rag-chat/
├── llm_client.py           # OpenAI LLM integration
├── rag_client.py           # RAG system with ChromaDB
├── embedding_pipeline.py   # Document processing pipeline
├── ragas_evaluator.py      # RAGAS evaluation metrics
├── chat.py                 # Streamlit chat application
├── requirements.txt        # Python dependencies
├── evaluation_dataset.txt  # Test questions for evaluation
├── README.md              # This file
├── apollo11/              # Apollo 11 mission documents
├── apollo13/              # Apollo 13 mission documents
└── challenger/            # Challenger mission documents
```

## Prerequisites

- Python 3.10 or higher
- OpenAI API key

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd nasa-rag-chat
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Step 1: Process Documents (Embedding Pipeline)

First, run the embedding pipeline to process NASA documents and create the vector database:

```bash
python embedding_pipeline.py --openai-key YOUR_API_KEY --data-path .
```

**Command-line options:**
- `--openai-key`: Your OpenAI API key (required)
- `--data-path`: Path to data directories (default: current directory)
- `--chroma-dir`: ChromaDB persist directory (default: `./chroma_db_openai`)
- `--collection-name`: Collection name (default: `nasa_space_missions_text`)
- `--chunk-size`: Text chunk size (default: 500)
- `--chunk-overlap`: Overlap between chunks (default: 100)
- `--update-mode`: How to handle existing documents: `skip`, `update`, or `replace` (default: `skip`)
- `--stats-only`: Only show collection statistics
- `--test-query`: Run a test query after processing

**Example with test query:**
```bash
python embedding_pipeline.py --openai-key YOUR_API_KEY --test-query "What happened on Apollo 13?"
```

### Step 2: Launch the Chat Application

Run the Streamlit chat interface:

```bash
streamlit run chat.py
```

This will open a web browser with the chat interface where you can:
- Select a document collection
- Enter your OpenAI API key
- Choose the AI model (GPT-3.5 or GPT-4)
- Ask questions about NASA missions
- View real-time evaluation scores

### Step 3: Test with Sample Questions

Use questions from `evaluation_dataset.txt` to test the system:

- "What was the primary mission objective of Apollo 11?"
- "What problems did Apollo 13 encounter during its mission?"
- "What caused the Challenger disaster?"
- "Who were the crew members on Apollo 11?"

## Components

### LLM Client (`llm_client.py`)
Handles OpenAI API integration with:
- NASA expert system prompt
- Conversation history management
- Context-aware response generation

### RAG Client (`rag_client.py`)
Manages document retrieval:
- ChromaDB backend discovery
- Semantic search with mission filtering
- Context formatting for LLM

### Embedding Pipeline (`embedding_pipeline.py`)
Processes NASA documents:
- Text chunking with overlap
- OpenAI embedding generation
- ChromaDB collection management
- Batch processing with progress tracking

### RAGAS Evaluator (`ragas_evaluator.py`)
Evaluates response quality using:
- Response Relevancy
- Faithfulness
- BLEU Score
- ROUGE Score

### Chat Application (`chat.py`)
Streamlit-based interface featuring:
- Real-time chat interface
- Document collection selection
- Model selection (GPT-3.5, GPT-4)
- Live evaluation metrics display
- Conversation history

## Supported NASA Missions

1. **Apollo 11** (July 1969)
   - First Moon landing
   - Mission reports and transcripts

2. **Apollo 13** (April 1970)
   - "Successful failure" mission
   - Technical air-to-ground transcripts

3. **Space Shuttle Challenger** (January 1986)
   - STS-51-L disaster analysis
   - Investigation reports

## Evaluation Metrics

The system evaluates responses using RAGAS metrics:

| Metric | Description |
|--------|-------------|
| Response Relevancy | How relevant the answer is to the question |
| Faithfulness | How well the answer is grounded in the context |
| BLEU Score | N-gram overlap with reference text |
| ROUGE Score | Recall-oriented evaluation |

## Troubleshooting

### Common Issues

1. **ChromaDB not found**: Run the embedding pipeline first to create the database.

2. **API key errors**: Ensure your OpenAI API key is valid and has available credits.

3. **Import errors**: Install all dependencies with `pip install -r requirements.txt`.

4. **Memory issues**: Reduce `--batch-size` in the embedding pipeline.

### Checking Collection Status

View collection statistics:
```bash
python embedding_pipeline.py --openai-key YOUR_API_KEY --stats-only
```

## Implementation Report

### Challenges Faced

1. **Chunking Strategy**: Implemented sentence-boundary-aware chunking to preserve context across chunks with configurable overlap.

2. **RAGAS Integration**: Required careful handling of async operations and proper sample formatting for the RAGAS SDK.

3. **Context Management**: Balanced context length for LLM calls while maintaining relevant information.

### Key Design Decisions

1. **Modular Architecture**: Separated concerns into distinct modules for maintainability and testing.

2. **Persistent Storage**: Used ChromaDB's persistent client for database durability.

3. **Configurable Pipeline**: Command-line arguments allow customization without code changes.

### Testing Approach

- Unit tested each component independently
- Integration tested the complete RAG pipeline
- Verified with sample questions from different mission categories

### Future Improvements

- Add support for additional NASA missions
- Implement streaming responses
- Add conversation export functionality
- Enhance evaluation with custom metrics
