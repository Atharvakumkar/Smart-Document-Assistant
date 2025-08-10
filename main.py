
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATA_PATH = "data/"
PDF_FILENAME = "Java_unit1.pdf"
CHROMA_PATH = "chroma_db"

#1 --> Loading the documents provided locally
def load_documents():
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from {pdf_path}")
    return documents

#2 --> Splitting the documents into chunks by providing values of the chunk size, etc.
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits

#3 --> Convert the text into numeric vector values so that the LLM can understand
def get_embedding_function(model_name="nomic-embed-text"):
    # Ensure Ollama server is running (ollama serve)
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        print(f"Initialized Ollama embeddings with model: {model_name}")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        print("Make sure Ollama is running (ollama serve) and the model is available")
        raise

#4 --> Getting the vectors and inserting them into CHROMADB
def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print(f"Vector store initialized/loaded from: {persist_directory}")
    return vectorstore

#5 --> Indexing the documents created in CHROMADB for faster output
def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    print(f"Indexing {len(chunks)} chunks...")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Existing vector store found at {persist_directory}")
        response = input("Do you want to overwrite existing data? (y/n): ")
        if response.lower() != 'y':
            print("Loading existing vector store...")
            return get_vector_store(embedding_function, persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore

#6 --> Create a RAG Chain language and initialize the LLM
def create_rag_chain(vector_store, llm_model_name="qwen2.5:7b", context_window=8192):
    try:
        # Initialize the LLM
        llm = ChatOllama(
            model=llm_model_name,
            temperature=0,
            num_ctx=context_window  # IMPORTANT
        )
        print(f"Initialized ChatOllama with model: {llm_model_name}, context window: {context_window}")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Make sure Ollama is running and the model is available")
        raise

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )
    print("Retriever initialized.")

    # Define the prompt template
    template = """Answer the question based ONLY on the following context:

{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#The LLM asks the user question
def query_rag(chain, question):
    print("\nQuerying RAG chain...")
    print(f"Question: {question}")
    try:
        response = chain.invoke(question)
        print("\nResponse:")
        print(response)
        return response
    except Exception as e:
        print(f"Error during query: {e}")
        return None

#Main function the actual flow of the process
def main():
    #Takes input from the user and generates output
    try:

        print("Step 1: Loading documents...")
        docs = load_documents()

        print("\nStep 2: Splitting documents...")
        chunks = split_documents(docs)

        print("\nStep 3: Initializing embeddings...")
        embedding_function = get_embedding_function()  # Using Ollama nomic-embed-text

        print("\nStep 4: Setting up vector store...")
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
            print("Existing vector store found. Loading...")
            vector_store = get_vector_store(embedding_function)
        else:
            print("Creating new vector store...")
            vector_store = index_documents(chunks, embedding_function)

        print("\nStep 5: Creating RAG chain...")
        rag_chain = create_rag_chain(vector_store, llm_model_name="qwen2.5:7b")

        print("\nStep 6: Ready for queries!")
        print("Enter your questions (type 'quit' to exit):")

        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if question:
                query_rag(rag_chain, question)

    except Exception as e:
        print(f"An error occurred: {e}")
        return 1

    return 0


#Main execution
if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
