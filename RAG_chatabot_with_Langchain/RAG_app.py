####################################################################
#                         import
####################################################################

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any

# Import OpenRouter (via OpenAI-compatible interface)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.schema import Document

# text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Local embeddings (HuggingFace sentence transformers)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import streamlit
import streamlit as st

####################################################################
#              Config: LLM services, assistant language,...
####################################################################

# OpenRouter models - popular options
OPENROUTER_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3.1-70b-instruct",
]

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd'hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Contextual compression",
    "Vectorstore backed retriever",
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)
# Museum data path - try multiple locations
_script_dir = Path(__file__).resolve().parent
_possible_paths = [
    _script_dir.parent.parent.joinpath("musee.oeuvres1.json"),  # Root of project
    _script_dir.parent.joinpath("musee.oeuvres1.json"),  # Parent directory
    Path("/app/musee.oeuvres1.json"),  # Docker path
]
MUSEUM_DATA_PATH = None
for path in _possible_paths:
    if path.exists():
        MUSEUM_DATA_PATH = path
        break
if MUSEUM_DATA_PATH is None:
    # Default to parent of parent
    MUSEUM_DATA_PATH = _script_dir.parent.parent.joinpath("musee.oeuvres1.json")

# Ensure directories exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#            Create app interface with streamlit
####################################################################

st.set_page_config(
    page_title="Museum Chatbot - RAG",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Museum RAG Chatbot")

# Initialize session state
# Default API key - can be overridden by environment variable or user input
DEFAULT_API_KEY = "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"
if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", DEFAULT_API_KEY)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = OPENROUTER_MODELS[0]

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5

if "top_p" not in st.session_state:
    st.session_state.top_p = 0.95

if "assistant_language" not in st.session_state:
    st.session_state.assistant_language = "french"

if "retriever_type" not in st.session_state:
    st.session_state.retriever_type = list_retriever_types[0]

if "error_message" not in st.session_state:
    st.session_state.error_message = ""


def load_museum_json(json_path: Path) -> List[Document]:
    """Load museum artwork data from JSON file and convert to LangChain documents."""
    documents = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # Create a comprehensive text representation of each artwork
            content_parts = []
            
            if "titre" in item:
                content_parts.append(f"Titre: {item['titre']}")
            
            if "description" in item:
                content_parts.append(f"Description: {item['description']}")
            
            if "inscription" in item:
                content_parts.append(f"Inscription: {item['inscription']}")
            
            if "periode" in item:
                content_parts.append(f"Période: {item['periode']}")
            
            if "categorie" in item:
                content_parts.append(f"Catégorie: {item['categorie']}")
            
            if "salle" in item:
                content_parts.append(f"Salle: {item['salle']}")
            
            if "lieu_decouverte" in item:
                if isinstance(item["lieu_decouverte"], dict):
                    lieu_info = ", ".join([f"{k}: {v}" for k, v in item["lieu_decouverte"].items() if isinstance(v, str)])
                    if lieu_info:
                        content_parts.append(f"Lieu de découverte: {lieu_info}")
                else:
                    content_parts.append(f"Lieu de découverte: {item['lieu_decouverte']}")
            
            if "dimensions" in item:
                dims = json.dumps(item["dimensions"], ensure_ascii=False)
                content_parts.append(f"Dimensions: {dims}")
            
            if "materiaux" in item:
                content_parts.append(f"Matériaux: {', '.join(item['materiaux'])}")
            
            if "decor" in item:
                if isinstance(item["decor"], dict):
                    decor_info = json.dumps(item["decor"], ensure_ascii=False)
                    content_parts.append(f"Décor: {decor_info}")
                else:
                    content_parts.append(f"Décor: {item['decor']}")
            
            if "motifs" in item:
                content_parts.append(f"Motifs: {', '.join(item['motifs'])}")
            
            if "disposition" in item:
                content_parts.append(f"Disposition: {item['disposition']}")
            
            if "interpretation" in item:
                content_parts.append(f"Interprétation: {item['interpretation']}")
            
            if "contexte_mythologique" in item:
                content_parts.append(f"Contexte mythologique: {item['contexte_mythologique']}")
            
            content = "\n".join(content_parts)
            
            # Create metadata
            metadata = {
                "source": str(json_path),
                "titre": item.get("titre", "Unknown"),
                "categorie": item.get("categorie", "Unknown"),
                "periode": item.get("periode", "Unknown"),
            }
            
            if "_id" in item:
                metadata["artwork_id"] = str(item["_id"])
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        st.success(f"✅ Loaded {len(documents)} artworks from museum database")
        return documents
    
    except FileNotFoundError:
        st.error(f"❌ Museum data file not found at {json_path}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"❌ Error parsing JSON: {e}")
        return []
    except Exception as e:
        st.error(f"❌ Error loading museum data: {e}")
        return []


def sidebar_configuration():
    """Create the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        st.subheader("OpenRouter API Key")
        api_key = st.text_input(
            "API Key",
            value=st.session_state.openrouter_api_key,
            type="password",
            placeholder="sk-or-v1-...",
            help="Get your API key from https://openrouter.ai"
        )
        st.session_state.openrouter_api_key = api_key
        
        st.divider()
        
        # Model selection
        st.subheader("Model Settings")
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            OPENROUTER_MODELS,
            help="Choose an AI model from OpenRouter"
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Controls randomness in responses"
        )
        
        st.session_state.top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Controls diversity via nucleus sampling"
        )
        
        st.divider()
        
        # Assistant language
        st.subheader("Language Settings")
        st.session_state.assistant_language = st.selectbox(
            "Assistant Language",
            list(dict_welcome_message.keys()),
            index=list(dict_welcome_message.keys()).index(st.session_state.assistant_language) if st.session_state.assistant_language in dict_welcome_message else 0
        )
        
        st.divider()
        
        # Retriever type
        st.subheader("Retrieval Settings")
        st.session_state.retriever_type = st.selectbox(
            "Retriever Type",
            list_retriever_types
        )
        
        st.divider()
        
        st.info("💡 **Note:** Your API key and settings are only used when creating or loading a vectorstore.")


def create_vectorstore_from_museum_data():
    """Create or load vectorstore from museum JSON data."""
    tab_create, tab_load = st.tabs(["Create Vectorstore", "Load Vectorstore"])
    
    with tab_create:
        st.subheader("Create New Vectorstore from Museum Data")
        
        vector_store_name = st.text_input(
            "Vectorstore Name",
            value="museum_artworks",
            placeholder="Enter a name for your vectorstore"
        )
        
        if st.button("Create Vectorstore", type="primary"):
            if not st.session_state.openrouter_api_key:
                st.error("❌ Please enter your OpenRouter API key in the sidebar.")
                return
            
            if not vector_store_name:
                st.error("❌ Please enter a vectorstore name.")
                return
            
            if not MUSEUM_DATA_PATH.exists():
                st.error(f"❌ Museum data file not found at {MUSEUM_DATA_PATH}")
                return
            
            with st.spinner("Creating vectorstore from museum data..."):
                try:
                    # 1. Load JSON documents
                    documents = load_museum_json(MUSEUM_DATA_PATH)
                    
                    if not documents:
                        st.error("❌ No documents loaded. Please check the JSON file.")
                        return
                    
                    # 2. Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1600,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_documents(documents)
                    st.info(f"📊 Split into {len(chunks)} chunks")
                    
                    # 3. Create embeddings (local)
                    embeddings = get_embeddings()
                    
                    # 4. Create vectorstore
                    persist_directory = LOCAL_VECTOR_STORE_DIR / vector_store_name
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=str(persist_directory)
                    )
                    
                    st.success(f"✅ Vectorstore '{vector_store_name}' created with {vector_store._collection.count()} chunks")
                    
                    # 5. Create retriever and chain
                    create_chain_from_vectorstore(vector_store, embeddings, vector_store_name)
                    
                except Exception as e:
                    st.error(f"❌ Error creating vectorstore: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab_load:
        st.subheader("Load Existing Vectorstore")
        
        # List available vectorstores
        if LOCAL_VECTOR_STORE_DIR.exists():
            vectorstores = [d.name for d in LOCAL_VECTOR_STORE_DIR.iterdir() if d.is_dir()]
        else:
            vectorstores = []
        
        if vectorstores:
            selected_vectorstore = st.selectbox(
                "Select Vectorstore",
                vectorstores
            )
            
            if st.button("Load Vectorstore", type="primary"):
                if not st.session_state.openrouter_api_key:
                    st.error("❌ Please enter your OpenRouter API key in the sidebar.")
                    return
                
                with st.spinner("Loading vectorstore..."):
                    try:
                        embeddings = get_embeddings()
                        vector_store = Chroma(
                            embedding_function=embeddings,
                            persist_directory=str(LOCAL_VECTOR_STORE_DIR / selected_vectorstore)
                        )
                        
                        st.success(f"✅ Vectorstore '{selected_vectorstore}' loaded with {vector_store._collection.count()} chunks")
                        
                        create_chain_from_vectorstore(vector_store, embeddings, selected_vectorstore)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading vectorstore: {e}")
        else:
            st.info("No vectorstores found. Create one first!")


def get_embeddings():
    """Get local HuggingFace embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )


def create_retriever(vector_store, embeddings, retriever_type="Vectorstore backed retriever"):
    """Create a retriever from vectorstore."""
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 16}
    )
    
    if retriever_type == "Vectorstore backed retriever":
        return base_retriever
    
    elif retriever_type == "Contextual compression":
        # Create compression retriever
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=16)
        reordering = LongContextReorder()
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter, reordering]
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    return base_retriever


def create_chain_from_vectorstore(vector_store, embeddings, vectorstore_name):
    """Create the RAG chain from vectorstore."""
    try:
        # Create retriever
        retriever = create_retriever(
            vector_store,
            embeddings,
            st.session_state.retriever_type
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )
        
        # Create LLM with OpenRouter
        standalone_query_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.session_state.openrouter_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",  # Optional
                "X-Title": "Museum Chatbot"  # Optional
            }
        )
        
        response_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.session_state.openrouter_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Museum Chatbot"
            }
        )
        
        # Create prompts
        condense_question_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""
        )
        
        answer_template_str = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in {st.session_state.assistant_language}.

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {st.session_state.assistant_language}."""
        
        answer_prompt = ChatPromptTemplate.from_template(answer_template_str)
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": answer_prompt},
            condense_question_llm=standalone_query_llm,
            llm=response_llm,
            memory=memory,
            retriever=retriever,
            chain_type="stuff",
            verbose=False,
            return_source_documents=True
        )
        
        # Store in session state
        st.session_state.chain = chain
        st.session_state.memory = memory
        st.session_state.vectorstore_name = vectorstore_name
        
        # Clear chat history
        clear_chat_history()
        
        st.success("✅ RAG chain created successfully! You can now chat.")
        
    except Exception as e:
        st.error(f"❌ Error creating chain: {e}")
        import traceback
        st.code(traceback.format_exc())


def clear_chat_history():
    """Clear chat history and memory."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    
    try:
        if "memory" in st.session_state:
            st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt: str):
    """Invoke the LLM and display results."""
    try:
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]
        
        # Handle string vs object response
        if hasattr(answer, 'content'):
            answer = answer.content
        
        # Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # Display source documents
            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(response["source_documents"], 1):
                    metadata = doc.metadata
                    titre = metadata.get("titre", "Unknown")
                    categorie = metadata.get("categorie", "")
                    
                    st.markdown(f"**Source {i}: {titre}**")
                    if categorie:
                        st.caption(f"Category: {categorie}")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.divider()
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())


def main():
    """Main chatbot interface."""
    sidebar_configuration()
    
    st.divider()
    
    # Vectorstore management
    create_vectorstore_from_museum_data()
    
    st.divider()
    
    # Chat interface
    col1, col2 = st.columns([7, 1])
    with col1:
        st.subheader("💬 Chat with Museum Data")
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()
    
    # Initialize messages
    if "messages" not in st.session_state:
        clear_chat_history()
    
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the museum artworks..."):
        if "chain" not in st.session_state:
            st.warning("⚠️ Please create or load a vectorstore first!")
            st.stop()
        
        if not st.session_state.openrouter_api_key:
            st.warning("⚠️ Please enter your OpenRouter API key in the sidebar!")
            st.stop()
        
        with st.spinner("🤔 Thinking..."):
            get_response_from_LLM(prompt)


if __name__ == "__main__":
    main()
