import streamlit as st
import hashlib
import shutil
from openai import OpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List
import numpy as np
from datetime import datetime
import logging
import json
import re
import asyncio
import os

load_dotenv()

# Disable HuggingFace Tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure an event loop exists to fix "no running event loop" error
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAFT Chatbot", page_icon="🤖", layout="wide")

# LLM backend. Any OpenAI-compatible endpoint works -- Ollama (local), OpenRouter,
# Together, Groq -- so switching provider is three env vars, not a code change.
# See .env.example. Defaults point at a local Ollama.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1-8b-ctx8k:latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))


class ChatLLM:
    """Thin wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        self.model = model or LLM_MODEL
        self.temperature = LLM_TEMPERATURE if temperature is None else temperature
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop,
        )
        return response.choices[0].message.content


logger.info(f"LLM backend: {LLM_BASE_URL} model={LLM_MODEL}")
llm = ChatLLM()

# Load the RAFT dataset as CHUNKS.
#
# This used to merge the chunks back into whole documents by doc_id before
# indexing, which meant the vector store held ~95 article-level vectors instead
# of 680 chunk-level ones -- and therefore that changing the chunking strategy
# had no effect on system behaviour at all. The merge itself is not wrong, it
# was in the wrong stage: see merge_chunks_by_doc_id() below.
def load_raft_dataset(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raft_data = json.load(f)
        logger.info(f"Loaded {len(raft_data)} chunks from {json_file_path}")
        return raft_data
    except Exception as e:
        logger.error(f"Failed to load RAFT dataset: {e}")
        return []


# The merge that used to run at index time, kept for the generation stage.
# Retrieve small chunks, then expand to the full article before building the
# LLM context -- this is the "parent document" baseline planned for Phase 2.
# Not wired in yet; indexing must be chunk-level first.
def merge_chunks_by_doc_id(raft_data):
    try:
        merged_docs = {}
        for doc in raft_data:
            doc_id = doc["doc_id"]
            if doc_id not in merged_docs:
                merged_docs[doc_id] = {
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "author": doc["author"],
                    "date": doc["date"],
                    "source": doc["source"],
                    "content": [],
                    "chunk_indices": []
                }
            merged_docs[doc_id]["content"].append(doc["content"])
            merged_docs[doc_id]["chunk_indices"].append(doc["chunk_index"])

        merged_data = []
        for doc_id, doc_info in merged_docs.items():
            sorted_chunks = sorted(
                zip(doc_info["content"], doc_info["chunk_indices"]),
                key=lambda x: x[1]
            )
            merged_content = "\n".join(chunk[0] for chunk in sorted_chunks)
            merged_data.append({
                "doc_id": doc_id,
                "title": doc_info["title"],
                "author": doc_info["author"],
                "date": doc_info["date"],
                "source": doc_info["source"],
                "content": merged_content,
                "chunk_indices": sorted(doc_info["chunk_indices"])
            })

        logger.info(f"Merged {len(raft_data)} chunks into {len(merged_data)} documents by doc_id")
        return merged_data
    except Exception as e:
        logger.error(f"Failed to merge chunks by doc_id: {e}")
        return []

# Cache embeddings model
@st.cache_resource
def get_embeddings_model():
    try:
        import numpy
        logger.info(f"NumPy 版本: {numpy.__version__}, 路径: {numpy.__file__}")
        
        import torch
        logger.info(f"PyTorch 版本: {torch.__version__}, 路径: {torch.__file__}")
        
        import transformers
        logger.info(f"Transformers 版本: {transformers.__version__}, 路径: {transformers.__file__}")
        
        import sentence_transformers
        logger.info(f"Sentence-Transformers 版本: {sentence_transformers.__version__}, 路径: {sentence_transformers.__file__}")
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MPNet-base-v2")
        logger.info("成功初始化 HuggingFaceEmbeddings")
        return embeddings
    except ImportError as e:
        logger.error(f"导入错误: {e}")
        st.error(f"导入错误: {e}。请检查依赖项安装。")
        raise
    except Exception as e:
        logger.error(f"无法初始化 HuggingFaceEmbeddings: {e}")
        st.error(f"初始化嵌入模型出错: {e}")
        raise

PERSIST_DIRECTORY = "./chroma_db_new"
CORPUS_FILE = "raft_documents.json"
# Bump when the indexing scheme changes (fields embedded, metadata shape, ...)
# so an existing store is treated as stale and rebuilt.
INDEX_SCHEMA_VERSION = "2-chunk-level"


def _corpus_fingerprint(raft_data):
    """Identity of what is indexed, so a stale store can be detected."""
    h = hashlib.sha256()
    h.update(INDEX_SCHEMA_VERSION.encode())
    h.update(str(len(raft_data)).encode())
    for doc in raft_data:
        h.update(doc["id"].encode())
        h.update(doc["content"].encode())
    return h.hexdigest()


# Load vector database.
#
# This used to call Chroma.from_texts() unconditionally on every startup, which
# APPENDS to the existing collection rather than loading it -- the store grew by
# one full copy of the corpus per boot (measured: 815 -> 910 rows, +95 each time,
# 910 rows holding only 95 unique documents). Duplicates then crowded out top-k:
# a k=10 search returned only 4 distinct articles. Now the persisted store is
# loaded when its fingerprint matches, and rebuilt only when absent or stale.
@st.cache_resource
def load_vector_db():
    raft_data = load_raft_dataset(CORPUS_FILE)
    if not raft_data:
        logger.error("No RAFT data loaded, vector database initialization failed.")
        return None

    for doc in raft_data:
        if "content" not in doc:
            logger.error(f"Chunk {doc.get('id')} is missing 'content' field")
            return None

    fingerprint = _corpus_fingerprint(raft_data)
    fingerprint_path = os.path.join(PERSIST_DIRECTORY, "corpus_fingerprint.txt")
    embeddings = get_embeddings_model()

    try:
        stored = None
        if os.path.exists(fingerprint_path):
            with open(fingerprint_path, encoding="utf-8") as f:
                stored = f.read().strip()

        if stored == fingerprint:
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
            )
            count = vectorstore._collection.count()
            if count == len(raft_data):
                logger.info(f"Loaded persisted vector store: {count} chunks (fingerprint match)")
                return vectorstore
            logger.warning(f"Fingerprint matched but count is {count}, expected {len(raft_data)}; rebuilding")
        else:
            logger.info("Vector store absent or stale; rebuilding")

        # Rebuild from scratch. delete_collection() is not enough: Chroma soft-deletes,
        # leaving the old segment's rows orphaned in the SQLite file, so every rebuild
        # would grow it. Remove the directory instead -- it is a generated artifact,
        # rebuilt from CORPUS_FILE in seconds. Guarded so it only ever removes something
        # that actually looks like a Chroma store.
        if os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")):
            shutil.rmtree(PERSIST_DIRECTORY)
            logger.info(f"Removed stale vector store at {PERSIST_DIRECTORY}")

        # Title is prepended per CHUNK. It used to be prepended once per merged
        # document, so most chunks carried no title signal at all.
        texts = [f"{doc['title']}\n{doc['content']}" for doc in raft_data]
        metadatas = [
            {
                "id": doc["id"],
                "doc_id": doc["doc_id"],
                "chunk_index": doc["chunk_index"],
                "title": doc["title"],
                "date": doc["date"],
                "content": doc["content"],
            }
            for doc in raft_data
        ]

        vectorstore = Chroma.from_texts(
            texts, embeddings, metadatas=metadatas, persist_directory=PERSIST_DIRECTORY
        )
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            f.write(fingerprint)
        logger.info(f"Built vector store: {vectorstore._collection.count()} chunks")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to load vector database: {e}")
        st.error(f"Failed to load vector database: {e}")
        return None

vector_db = load_vector_db()
if vector_db is None:
    st.error("Failed to load vector database. Please check the RAFT dataset file and dependencies.")
    st.stop()

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Extract keywords from query using Prompt Engineering
def extract_keywords(question):
    try:
        prompt = """
        Given the following user question: "{question}"

        Your task is to extract all relevant keywords from the question, including person names, locations, events, or topics. Return the keywords as a comma-separated list (e.g., "Li Ming, Beijing, weather"). If no clear keywords are found, return an empty string "".

        Examples:
        - Question: "Recent news about Li Ming in Beijing"
          Answer: "Li Ming, Beijing, recent news"
        - Question: "Who won the 2024 Olympics in Paris?"
          Answer: "2024 Olympics, Paris, winner"
        - Question: "How's the weather in Tokyo today?"
          Answer: "Tokyo, weather, today"
        - Question: "General knowledge question"
          Answer: ""

        Return only the extracted keywords as a comma-separated list (or an empty string), without any explanation or additional text.
        """
        keywords_str = llm._call(prompt.format(question=question)).strip()
        
        # If an empty string is returned, it means no keywords were found, return an empty list
        if not keywords_str:
            logger.debug(f"No keywords found in question: '{question}'")
            return []
        
        # Split the comma-separated string into a list of keywords
        keywords = [keyword.strip() for keyword in keywords_str.split(",")]
        logger.debug(f"Extracted keywords: {keywords} from question: '{question}'")
        return keywords

    except Exception as e:
        logger.error(f"Error extracting keywords using LLM: {e}")
        return []

# Keyword-based search function (with a limit of 10 documents)
def keyword_search(vectorstore, keywords):
    if not keywords:
        logger.warning("No keywords provided for search.")
        return []

    logger.info(f"Performing keyword-based search with keywords: {keywords}")
    
    # Retrieve all documents from the vectorstore
    all_docs = vectorstore._collection.get(include=["metadatas", "documents"])["metadatas"]
    
    if not all_docs:
        logger.warning("No documents found in the vectorstore.")
        return []

    # Filter documents that contain any of the keywords
    matched_docs = []
    for doc in all_docs:
        doc_content = doc.get("content", "").lower()
        if any(keyword.lower() in doc_content for keyword in keywords):
            matched_docs.append(doc)

    # Convert matched documents into the same format as retriever output for consistency
    matched_docs_formatted = []
    for doc in matched_docs:
        # Create a dummy LangChain Document object for compatibility with downstream code
        from langchain_core.documents import Document
        formatted_doc = Document(
            page_content=f"Title: {doc['title']}\nContent: {doc['content']}",
            metadata=doc
        )
        matched_docs_formatted.append(formatted_doc)

    # Sort documents by the number of keyword matches (more matches = higher rank)
    matched_docs_sorted = sorted(
        matched_docs_formatted,
        key=lambda doc: sum(keyword.lower() in doc.metadata.get("content", "").lower() for keyword in keywords),
        reverse=True
    )

    # Limit to the top 10 documents
    max_docs = 10
    limited_docs = matched_docs_sorted[:max_docs]
    logger.info(f"Keyword search retrieved {len(matched_docs_sorted)} documents, limited to {len(limited_docs)}: {[doc.page_content[:50] for doc in limited_docs]}")
    return limited_docs

# RAFT question-answering logic
def ask_raft(question, vectorstore):
    messages = st.session_state.sessions.get(st.session_state.current_session, [])
    conversation_history = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Extract the most recent assistant response (if any)
    last_assistant_response = None
    for message in reversed(messages):
        if message["role"] == "assistant":
            last_assistant_response = message["content"]
            break
    logger.info(f"Most recent assistant response: '{last_assistant_response}'")

    no_answer_phrases = [
        "i don't", "i have no information", "i'm not sure", "i cannot provide", "up-to-date", "I'm not aware",
        "i do not have the answer", "unable to find", "no relevant information", "cutoff",
        "not mentioned", "does not appear", "dosn't mention", "provided context", "context provided",
        "The new context provided does not relate to the original question", "cannot provide", "I don't know", "no relevant information",
        "cannot find", "not mentioned", "not sure", "cannot find relevant content", "cannot", "knowledge cutoff", "irrelevant",
        "not possible to determine", "absence of relevant information", "do not contain any information",
        "no relevant content", "does not apply to the question", "exclusively discuss",
        "i am unable to", "i lack the information", "i have no knowledge", "i am unaware",
        "no information available", "information is missing", "cannot be determined",
        "not found in the context", "not specified", "not available in the data",
        "beyond my knowledge", "outside my knowledge", "not within my knowledge",
        "no data available", "data is insufficient", "insufficient information",
        "not covered in the documents", "not present in the documents",
        "no record of", "no mention of", "lacking details", "details are missing",
        "not enough context", "context is insufficient", "context does not contain",
        "not relevant to the question", "irrelevant to the query", "unrelated to the question", "cannot determine",
        "cannot answer", "unable to answer because", "answer is unavailable",
        "information not provided", "not included in the context", "not part of the data","i do not have",
    ]
    
    logger.info(f"Step 1: Start processing question: '{question}'")
    logger.info(f"Conversation history: {conversation_history}")
    logger.info(f"Current date: {current_date}")

    # Step 1.1: Refine the question into a retrieval query
    logger.info("Step 1.1: Refine the question for retrieval")
    refine_query_prompt = """
    Given the user question: "{question}"
    Current date: {current_date}

    Generate a concise search query (less than 10 words) to retrieve the most relevant information from a document database. Focus on extracting core keywords, removing unnecessary words (e.g., "please", "tell me"). If applicable, include time-sensitive terms (e.g., "recent", "2025"). Do not include explanations, only provide the refined query.
    """
    refined_query = llm._call(refine_query_prompt.format(
        question=question,
        current_date=current_date
    )).strip()
    logger.info(f"Step 1.2: Refined retrieval query: '{refined_query}'")

    # Step 1.3: Extract keywords
    keywords = extract_keywords(question)
    logger.info(f"Step 1.3: Extracted keywords: {keywords}")

    # Step 2: Keyword-based search
    logger.info("Step 2: Perform keyword-based search")
    initial_docs = keyword_search(vectorstore, keywords)
    if not initial_docs:
        logger.warning("No initial documents retrieved from keyword search.")
    else:
        logger.info(f"Retrieved {len(initial_docs)} documents: {[doc.page_content[:100] for doc in initial_docs]}")
        for doc in initial_docs:
            logger.debug(f"Initial retrieved document metadata: {doc.metadata}")

    # Step 3: Check document relevance (simplified, since we're using keyword matching)
    # def is_relevant_docs(docs, question):
    #     embeddings = get_embeddings_model()
    #     question_embedding = embeddings.embed_query(question)
    #     for doc in docs:
    #         doc_embedding = embeddings.embed_query(doc.page_content)
    #         similarity = np.dot(question_embedding, doc_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(doc_embedding))
    #         logger.debug(f"Similarity with doc '{doc.page_content[:50]}...': {similarity}")
    #         if similarity > 0.5:
    #             return True
    #     return False
    def is_relevant_docs(docs, question):
        # Since we're using keyword matching, assume docs are relevant if they were retrieved
        return bool(docs)

    initial_relevant = is_relevant_docs(initial_docs, question) if initial_docs else False
    logger.info(f"Step 3: Are initial documents relevant to the question? {initial_relevant}")

    # Step 4: If relevant documents are retrieved, attempt to extract an answer
    if initial_relevant:
        logger.info("Step 4: Found relevant documents, attempting to extract an answer")
        def format_doc_content(doc):
            metadata = doc.metadata
            title = metadata.get('title', 'Unknown Title')
            content = metadata.get('content', 'Content Unavailable')
            return f"Title: {title}\nContent: {content}"

        initial_docs_content = [format_doc_content(doc) for doc in initial_docs]
        initial_answer_prompt = """
        Using the following documents, answer the user's question. Provide only the direct answer, without any reasoning, explanation, or thought process. If the answer cannot be determined, explicitly state: "As of {current_date}, I do not have sufficient information to determine the answer to '{question}'."

        User question: {question}

        Documents:
        {docs_content}

        Provide only the direct answer.
        """
        initial_answer = llm._call(initial_answer_prompt.format(
            question=question,
            current_date=current_date,
            docs_content='\n'.join(initial_docs_content)
        )).strip()
        logger.info(f"Step 4.1: Initial answer extracted from documents: '{initial_answer}'")

        # If the initial answer is sufficient, return it directly
        if initial_answer and not any(phrase in initial_answer.lower() for phrase in no_answer_phrases):
            logger.info("Step 4.2: Initial answer is sufficient, returning directly")
            return initial_answer
        else:
            logger.info("Step 4.2: Initial answer is insufficient, proceeding to the RAFT path")

    # Step 5: Determine if the question depends on previous responses
    history_dependent_keywords = [
        "previous answer", "last response", "earlier question", "just now",
    ]

    history_dependent_patterns = [
        r"what\s+(did\s+you|was\s+the)\s+(say|mention|answer|response)",
        r"the\s+(previous|last|earlier)\s+(answer|response|thing)",
        r"(can|could)\s+you\s+(repeat|say\s+again)",
        r"(repeat|restate)\s+(that|the\s+answer)",
    ]

    is_history_dependent = any(keyword in question.lower() for keyword in history_dependent_keywords)
    if not is_history_dependent:
        is_history_dependent = any(re.search(pattern, question.lower()) for pattern in history_dependent_patterns)

    if not is_history_dependent:
        history_dependent_prompt = f"""
        Determine if the following question explicitly references a previous answer or response in the conversation history.
        Question: "{question}"

        If the question depends on the conversation history (e.g., asking about a previous answer or what was just said), answer 'Yes'.
        If the question does not depend on the conversation history (e.g., an independent question like "What is 1+1?"), answer 'No'.
        Provide only the answer ('Yes' or 'No'), without any reasoning.
        """
        history_dependent_result = llm._call(history_dependent_prompt).strip().lower()
        is_history_dependent = history_dependent_result == "yes"
    logger.info(f"Step 5: Does the question depend on history? {is_history_dependent}")

    # Step 6: Retrieve documents for the RAFT prompt
    logger.info("Step 6: Retrieve relevant documents using keyword search")
    retrieved_docs = keyword_search(vectorstore, keywords)
    if not retrieved_docs:
        logger.warning("No documents retrieved from keyword search.")
    else:
        logger.info(f"Retrieved {len(retrieved_docs)} documents: {[doc.page_content[:100] for doc in retrieved_docs]}")
        for doc in retrieved_docs:
            logger.debug(f"Retrieved document metadata: {doc.metadata}")

    # Step 7: Check document relevance
    relevant = is_relevant_docs(retrieved_docs, question) if retrieved_docs else False
    logger.info(f"Step 7: Are documents relevant to the question? {relevant}")

    # Step 8: If no relevant documents, fall back to the model's general knowledge
    if not retrieved_docs or not relevant:
        logger.info("Step 8: No relevant documents found, falling back to general knowledge")
        
        # If the question depends on history, use a specific prompt
        if is_history_dependent:
            llm_prompt = """
            Conversation history: {conversation_history}
            Most recent assistant response: {last_assistant_response}
            User question: {question}
            Current date: {current_date}

            The question references a previous answer or recent response. Use the most recent assistant response provided above to answer the question. If the most recent assistant response is 'None' or does not contain the information needed to answer the question, return: "Due to insufficient conversation history, I cannot determine the previous answer to '{question}'."
            Provide only the direct answer, without any reasoning, explanation, or thought process.
            """
            llm_prompt = llm_prompt.format(
                conversation_history=conversation_history,
                last_assistant_response=last_assistant_response if last_assistant_response else 'None',
                question=question,
                current_date=current_date
            )
        else:
            # For independent questions not relying on history, answer directly
            llm_prompt = """
            User question: {question}
            Current date: {current_date}

            Answer the question directly using your general knowledge. Provide only the direct answer, without any reasoning, explanation, or thought process.
            """
            llm_prompt = llm_prompt.format(question=question, current_date=current_date)

        llm_answer = llm._call(llm_prompt).strip()
        logger.info(f"LLM answer (no relevant documents): '{llm_answer}'")
        
        if not llm_answer or any(phrase in llm_answer.lower() for phrase in no_answer_phrases):
            logger.info("Step 8.1: LLM answer is insufficient and there is no corpus match")
            return f"As of {current_date}, I do not have sufficient information to answer '{question}'."
        logger.info("Step 8.1: LLM answer is sufficient, returning directly")
        return llm_answer

    # Step 9: RAFT logic
    logger.info("Step 9: Documents are relevant, proceeding with RAFT logic")
    mid_point = len(retrieved_docs) // 2
    golden_docs = retrieved_docs[:mid_point]
    distractor_docs = retrieved_docs[mid_point:]
    
    def format_doc_content(doc):
        metadata = doc.metadata
        title = metadata.get('title', 'Unknown Title')
        content = metadata.get('content', 'Content Unavailable')
        return f"Title: {title}\nContent: {content}"

    golden_docs_content = [format_doc_content(doc) for doc in golden_docs]
    distractor_docs_content = [format_doc_content(doc) for doc in distractor_docs]
    
    # If the question depends on history, include the most recent assistant response in the RAFT prompt
    if is_history_dependent:
        raft_prompt = """
        You are a model trained with RAFT (Retrieval-Augmented Fine-Tuning), capable of extracting answers from provided documents while ignoring irrelevant information.
        The question references a previous answer or recent response. Use the most recent assistant response provided below to answer the question. If the most recent assistant response is 'None' or does not contain the required information, search for the answer in the "Golden Documents" and ignore the "Distractor Documents". If neither the documents nor the history provide sufficient information to answer the question, use your general knowledge based on the current date ({current_date}) to provide the most accurate answer. If you still cannot determine the answer, return: "Due to insufficient conversation history, I cannot determine the previous answer to '{question}'."

        User question: {question}
        Conversation history: {conversation_history}
        Most recent assistant response: {last_assistant_response}

        Golden Documents:
        {golden_docs_content}

        Distractor Documents:
        {distractor_docs_content}

        Provide only the direct answer, without any reasoning, explanation, or thought process.
        """
        raft_prompt = raft_prompt.format(
            current_date=current_date,
            question=question,
            conversation_history=conversation_history,
            last_assistant_response=last_assistant_response if last_assistant_response else 'None',
            golden_docs_content='\n'.join(golden_docs_content),
            distractor_docs_content='\n'.join(distractor_docs_content)
        )
    else:
        raft_prompt = """
        You are a model trained with RAFT (Retrieval-Augmented Fine-Tuning), capable of extracting answers from provided documents while ignoring irrelevant information.
        Search for the answer to the user's question in the "Golden Documents", ignoring the "Distractor Documents". If the documents do not provide sufficient information to answer the question, use your general knowledge based on the current date ({current_date}) to provide the most accurate answer. If you still cannot determine the answer, explicitly state: "As of {current_date}, I do not have sufficient information to determine the answer to '{question}'."

        User question: {question}

        Golden Documents:
        {golden_docs_content}

        Distractor Documents:
        {distractor_docs_content}

        Provide only the direct answer, without any reasoning, explanation, or thought process.
        """
        raft_prompt = raft_prompt.format(
            current_date=current_date,
            question=question,
            golden_docs_content='\n'.join(golden_docs_content),
            distractor_docs_content='\n'.join(distractor_docs_content)
        )

    raft_response = llm._call(raft_prompt).strip()
    logger.info(f"RAFT response: '{raft_response}'")

    return raft_response

# Initialize LangChain memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Streamlit UI section
st.markdown("""
    <style>
        .chat-container {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 8px;
            max-width: 80%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .user-container {
            display: flex;
            justify-content: flex-end !important;
            margin-left: auto !important;
            flex-direction: row-reverse;
            width: auto;
            min-width: 0;
            float: right;
        }
        .bot-container {
            display: flex;
            justify-content: flex-start !important;
            flex-direction: row;
            width: auto;
            min-width: 0;
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin: 0 10px;
            flex-shrink: 0;
        }
        .message {
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 70%;
            word-wrap: break-word;
            overflow-wrap: break-word;
            display: inline-block;
        }
        .user-message {
            background-color: #e0f7fa;
            color: black;
            text-align: right;
        }
        .bot-message {
            background-color: #f1f1f1;
            color: black;
            text-align: left;
        }
        .stTextInput > div > div > input {
            height: 50px;
            font-size: 16px;
        }
        .stMarkdown, .stMarkdown > div {
            width: 100% !important;
            max-width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session records
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.current_session = "Session 1"

# Sidebar: Historical sessions
with st.sidebar:
    st.header("Historical Sessions")
    for session_name in st.session_state.sessions:
        if st.button(f"{session_name}", key=session_name):
            st.session_state.current_session = session_name
            st.rerun()
    if st.button("Create New Session"):
        new_session_name = f"Session {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[new_session_name] = []
        st.session_state.current_session = new_session_name
        st.rerun()

# Main interface
st.title("🤖 RAFT Chatbot")
st.subheader(f"Current Session: {st.session_state.current_session}")

# Display conversation history
st.write("### Conversation History")
messages = st.session_state.sessions.get(st.session_state.current_session, [])
for message in messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="chat-container user-container">
                <img src="https://cdn-icons-png.flaticon.com/512/149/149071.png" class="avatar">
                <div class="message user-message">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-container bot-container">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712106.png" class="avatar">
                <div class="message bot-message">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)

# Input box and send button
if "user_message" not in st.session_state:
    st.session_state.user_message = ""

user_message = st.text_input("💬 Enter Your Question:", value="", key="user_message_input")
if st.button("Send"):
    if user_message:
        messages.append({"role": "user", "content": user_message})
        raft_response = ask_raft(user_message, vector_db)
        messages.append({"role": "assistant", "content": raft_response})
        st.session_state.sessions[st.session_state.current_session] = messages
        st.session_state.user_message = ""
        st.rerun()