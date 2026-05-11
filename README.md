# Smart Document Assistant

A privacy-first, fully offline Retrieval-Augmented Generation system that lets you query PDF documents through a natural language CLI. The system loads a PDF, splits it into chunks, embeds those chunks into a local vector store, and answers questions using a locally running LLM — with no API keys, no internet connection, and no data leaving the machine.

## Table of Contents

- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Vector Store Behaviour](#vector-store-behaviour)
- [Model Reference](#model-reference)
- [Author](#author)
- [License](#license)

---

## Architecture

The system is a standard RAG pipeline with all inference running locally via Ollama. There are two distinct phases:

**Indexing phase** (runs once per document or on overwrite)

```
PDF file --> PyPDFLoader --> RecursiveCharacterTextSplitter --> OllamaEmbeddings (nomic-embed-text) --> ChromaDB (persisted to disk)
```

**Query phase** (runs on every question)

```
User question --> ChromaDB similarity search (k=3) --> Retrieved chunks + question --> ChatPromptTemplate --> ChatOllama (qwen2.5:7b) --> StrOutputParser --> Answer
```

---

## Pipeline

The `main()` function executes six sequential steps:

**Step 1 — Load documents**

`PyPDFLoader` reads the PDF from `data/` and returns one `Document` object per page, preserving page metadata.

**Step 2 — Split documents**

`RecursiveCharacterTextSplitter` breaks pages into overlapping chunks using the following parameters:

| Parameter | Value |
|---|---|
| chunk_size | 1000 characters |
| chunk_overlap | 200 characters |
| length_function | len |
| is_separator_regex | False |

**Step 3 — Generate embeddings**

`OllamaEmbeddings` uses the `nomic-embed-text` model running locally via Ollama to convert each chunk into a dense vector representation.

**Step 4 — Set up vector store**

ChromaDB is initialised with the embedding function and a local persist directory (`chroma_db/`). If an existing store is found, it is loaded. If not, a new one is created.

**Step 5 — Create RAG chain**

A LangChain LCEL chain is assembled:

```
{"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ChatPromptTemplate
    | ChatOllama (qwen2.5:7b)
    | StrOutputParser
```

The retriever uses similarity search with `k=3`, returning the three most relevant chunks per query. The prompt instructs the LLM to answer strictly from the retrieved context.

**Step 6 — Interactive query loop**

A `while True` loop reads questions from stdin, invokes the chain, and prints the response. Type `quit`, `exit`, or `q` to terminate.

---

## Project Structure

```
Smart-Document-Assistant/
|
|-- main.py           # Full RAG pipeline: loading, chunking, embedding, indexing, querying
|-- data/             # Place your PDF file(s) here
|-- chroma_db/        # Persisted ChromaDB vector store (auto-created on first run)
|-- .env              # Environment variables (optional; loaded via python-dotenv)
|-- .idea/            # PyCharm IDE configuration (not required to run)
```

---

## Tech Stack

**Language**

- Python 3.x

**RAG Framework**

- LangChain — document loading, text splitting, prompt templating, chain composition via LCEL

**LLM Runtime**

- Ollama — serves both the embedding model and the generative LLM locally

**Models**

- `nomic-embed-text` — embedding model for vector generation
- `qwen2.5:7b` — generative LLM for answer synthesis (temperature: 0, context window: 8192 tokens)

**Vector Store**

- ChromaDB — local, persistent vector database; stores and retrieves embeddings by cosine similarity

**PDF Parsing**

- `langchain_community.document_loaders.PyPDFLoader` — extracts text from PDF pages

**Output Parsing**

- `langchain_core.output_parsers.StrOutputParser` — strips the raw LLM response to plain text

---

## Prerequisites

- Python 3.9 or above
- [Ollama](https://ollama.com) installed and running
- The required models pulled into Ollama:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

- Ollama server running before starting the application:

```bash
ollama serve
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Atharvakumkar/Smart-Document-Assistant.git
cd Smart-Document-Assistant
```

Install Python dependencies:

```bash
pip install langchain langchain-community langchain-ollama langchain-core chromadb pypdf python-dotenv
```

---

## Configuration

The following constants at the top of `main.py` control the application's behaviour:

| Constant | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/` | Directory where input PDF files are stored |
| `PDF_FILENAME` | `Java_unit1.pdf` | Name of the PDF file to load |
| `CHROMA_PATH` | `chroma_db` | Directory where ChromaDB persists its data |

To use a different PDF, place the file in the `data/` directory and update `PDF_FILENAME`:

```python
PDF_FILENAME = "your_document.pdf"
```

To switch the LLM or embedding model, update the relevant arguments in `main()`:

```python
embedding_function = get_embedding_function(model_name="nomic-embed-text")
rag_chain = create_rag_chain(vector_store, llm_model_name="qwen2.5:7b", context_window=8192)
```

---

## Running the Application

Ensure Ollama is running, then start the application:

```bash
python main.py
```

On first run, the system will index the PDF and create the ChromaDB vector store. Subsequent runs detect the existing store and skip re-indexing unless you choose to overwrite. Once initialised, the CLI enters the interactive query loop:

```
Enter your questions (type 'quit' to exit):

Question: What is the difference between an abstract class and an interface?

Response:
...
```

---

## Vector Store Behaviour

ChromaDB is persisted to the `chroma_db/` directory after the first indexing run. On subsequent runs:

- If `chroma_db/` exists and is non-empty, the existing store is loaded and indexing is skipped.
- If `chroma_db/` does not exist or is empty, the full indexing pipeline runs.
- If `chroma_db/` exists but you want to re-index (e.g. after changing the PDF), the application prompts for confirmation before overwriting:

```
Existing vector store found at chroma_db
Do you want to overwrite existing data? (y/n):
```

---

## Model Reference

**nomic-embed-text**

Used for generating embeddings. Produces 768-dimensional dense vectors. Runs entirely on-device via Ollama. Required to be pulled before first use.

**qwen2.5:7b**

Used for answer generation. A 7-billion parameter instruction-tuned model from Alibaba. Temperature is set to 0 for deterministic, factual responses. Context window is set to 8192 tokens to accommodate retrieved chunks alongside the question. Runs entirely on-device via Ollama.

---

## Author

Atharva Kumkar

---

## License

This project is released under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, and to permit persons to whom the software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
