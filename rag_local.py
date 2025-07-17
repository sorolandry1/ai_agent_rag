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
PDF_FILENAME = "livre.pdf"
CHROMA_PATH = "chroma_db"

def load_documents():
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier PDF {pdf_path} est introuvable.")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"{len(documents)} page(s) chargée(s) depuis {pdf_path}")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"{len(all_splits)} chunks générés")
    return all_splits

def get_embedding_function(model_name="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Embeddings Ollama initialisés avec le modèle : {model_name}")
    return embeddings

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    if os.path.exists(os.path.join(persist_directory, "index")):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        print(f"Base vectorielle chargée depuis : {persist_directory}")
    else:
        vectorstore = None
        print(f"Aucune base vectorielle trouvée à : {persist_directory}")
    return vectorstore

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    print(f"Indexation de {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Indexation terminée. Données sauvegardées dans : {persist_directory}")
    return vectorstore

def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0,
        num_ctx=context_window
    )
    print(f"ChatOllama initialisé avec le modèle : {llm_model_name}, fenêtre de contexte : {context_window}")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )
    print("Retriever initialisé.")
    template = """Réponds à la question UNIQUEMENT avec le contexte suivant :
{context}

Question : {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template créé.")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Chaîne RAG créée.")
    return rag_chain

def query_rag(chain, question):
    print("\nQuestion :", question)
    try:
        response = chain.invoke(question)
        print("Réponse :\n", response)
    except Exception as e:
        print("Erreur lors de la requête :", e)

if __name__ == "__main__":
    try:
        docs = load_documents()
        chunks = split_documents(docs)
        embedding_function = get_embedding_function()
        vector_store = get_vector_store(embedding_function)
        if vector_store is None:
            vector_store = index_documents(chunks, embedding_function)
        else:
            print("Indexation sautée (base déjà existante).")
        rag_chain = create_rag_chain(vector_store, llm_model_name="qwen3:8b")
        query_rag(rag_chain, "Quel est le sujet principal du document ?")
        query_rag(rag_chain, "Résume la section introduction.")
    except Exception as e:
        print("Erreur critique :", e)