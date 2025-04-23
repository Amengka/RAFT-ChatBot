import streamlit as st
import together
from together import Together
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List, Mapping, Any
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
import numpy as np
from datetime import datetime
import logging
import json
import re
import asyncio
import os

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

# Delayed import of PyTorch-related modules to prevent Streamlit file watcher from scanning them
def initialize_chatbot():
    # Configure SerpAPI
    serp_search = SerpAPIWrapper(serpapi_api_key="678e395a6c7c95e1b135322d29b35b9e7fe14712eed8900c372a31622440fbeb")
    search_tool = Tool(name="Search", func=serp_search.run, description="Real-time internet search")
    return search_tool

# Set Together AI API key
together.api_key = "tgp_v1_BjABj4CPzcLjXO1_xh8yg0UFLoQ1cAVjpSYyozUNTNo"
st.set_page_config(page_title="RAFT Chatbot", page_icon="🤖", layout="wide")

# Custom LLM class
class TogetherLLM:
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def __init__(self):
        self.client = Together(api_key=together.api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "TogetherLLM"

# Load RAFT dataset and merge by doc_id
def load_raft_dataset(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raft_data = json.load(f)
        logger.info(f"Successfully loaded RAFT dataset from {json_file_path}")

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
        for doc in merged_data:
            logger.debug(f"Document {doc['doc_id']}: {doc}")
        return merged_data
    except Exception as e:
        logger.error(f"Failed to load RAFT dataset: {e}")
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

# Load vector database
@st.cache_resource
def load_vector_db():
    raft_data = load_raft_dataset("raft_documents.json")
    if not raft_data:
        logger.error("No RAFT data loaded, vector database initialization failed.")
        return None

    for doc in raft_data:
        if "content" not in doc:
            logger.error(f"Document {doc['doc_id']} is missing 'content' field: {doc}")
            return None

    texts = [f"{doc['title']}\n{doc['content']}" for doc in raft_data]
    metadatas = [
        {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "date": doc["date"],
            "content": doc["content"],
            "chunk_indices": ",".join(map(str, doc["chunk_indices"]))
        }
        for doc in raft_data
    ]

    for meta in metadatas:
        logger.debug(f"Metadata for document {meta['doc_id']}: {meta}")

    embeddings = get_embeddings_model()
    persist_directory = "./chroma_db_new"
    try:
        vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=persist_directory)
        vectorstore.persist()
        logger.info("Vector database loaded successfully.")
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

# Initialize search tool after Streamlit setup
search_tool = initialize_chatbot()

# Extract keywords from query using Prompt Engineering
def extract_keywords(question):
    try:
        llm = TogetherLLM()
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

    # Step 1.1: Refine the question to generate a search query (optional, keeping for consistency)
    logger.info("Step 1.1: Refine the question for retrieval")
    llm = TogetherLLM()
    refine_query_prompt = """
    Given the user question: "{question}"
    Current date: {current_date}

    Generate a concise search query (less than 10 words) to retrieve the most relevant information from a document database. Focus on extracting core keywords, removing unnecessary words (e.g., "please", "tell me"). If applicable, include time-sensitive terms (e.g., "recent", "2025"). Do not include explanations, only provide the refined query.
    """
    refined_query = llm._call(refine_query_prompt.format(
        question=question,
        current_date=current_date
    )).strip()
    logger.info(f"Step 1.2: Refined search query: '{refined_query}'")

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
        llm = TogetherLLM()
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
            logger.info("Step 4.2: Initial answer is insufficient, proceeding to next steps")

    # Step 5: Determine if the question is time-sensitive
    time_sensitive_keywords = [
        "recent", "latest", "current", "now", "today", "yesterday", "live",
        "upcoming", "next", "right now", "recently", "just happened",
    ]

    time_sensitive_patterns = [
        r"last\s+(week|month|year|season|event|weekend|night|morning|day|hour|minute)",
        r"this\s+(week|month|year|season|event|weekend|morning|day)",
        r"next\s+(week|month|year|season|event|weekend|day)",
        r"in\s+\d{4}",
        r"on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",
        r"\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"at\s+\d{1,2}:\d{2}",
        r"(today|yesterday|tomorrow)\s+at",
        r"\d{4}-\d{2}-\d{2}",
    ]

    is_time_sensitive = any(keyword in question.lower() for keyword in time_sensitive_keywords)
    if not is_time_sensitive:
        is_time_sensitive = any(re.search(pattern, question.lower()) for pattern in time_sensitive_patterns)

    if not is_time_sensitive:
        llm = TogetherLLM()
        time_sensitive_prompt = f"""
        Determine if the following question requires real-time or recent information to answer accurately.
        Question: "{question}"
        Current date: {current_date}

        If the question is time-sensitive (e.g., asking about recent events, current rankings, or upcoming events), answer 'Yes'.
        If the question is not time-sensitive (e.g., asking about historical facts or general knowledge), answer 'No'.
        Provide only the answer ('Yes' or 'No'), without any reasoning.
        """
        time_sensitive_result = llm._call(time_sensitive_prompt).strip().lower()
        is_time_sensitive = time_sensitive_result == "yes"
    logger.info(f"Step 5: Is the question time-sensitive? {is_time_sensitive}")

    # Step 6: Determine if the question depends on previous responses
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
        llm = TogetherLLM()
        history_dependent_prompt = f"""
        Determine if the following question explicitly references a previous answer or response in the conversation history.
        Question: "{question}"

        If the question depends on the conversation history (e.g., asking about a previous answer or what was just said), answer 'Yes'.
        If the question does not depend on the conversation history (e.g., an independent question like "What is 1+1?"), answer 'No'.
        Provide only the answer ('Yes' or 'No'), without any reasoning.
        """
        history_dependent_result = llm._call(history_dependent_prompt).strip().lower()
        is_history_dependent = history_dependent_result == "yes"
    logger.info(f"Step 6: Does the question depend on history? {is_history_dependent}")

    # Step 7: If the question is time-sensitive, prioritize using the search tool
    if is_time_sensitive:
        logger.info("Step 7: Detected a time-sensitive question, proceeding with search")
        llm = TogetherLLM()
        
        # Generate a search query
        search_query_prompt = """
        Given the user question: "{question}"
        Current date: {current_date}
        
        Generate a concise, natural search query to retrieve the most relevant and up-to-date information from the internet. Use key terms from the question, ensuring the query aligns with how information is presented online (e.g., for event locations, include "location" or "held"). If the question involves recent or upcoming events, include the current year (e.g., "2025") to focus on the latest events. Keep the query concise and clear to ensure search accuracy.
        """
        search_query = llm._call(search_query_prompt.format(question=question, current_date=current_date)).strip()
        logger.info(f"Step 7.1: Generated search query: '{search_query}'")
        
        # Perform the search
        try:
            search_results = search_tool.run(search_query)
            logger.info(f"Step 7.2: Search results: '{search_results}'")
            if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
                logger.warning(f"SerpAPI returned no results or an error for query '{search_query}'")
                fallback_query_prompt = """
                Given the user question: "{question}"
                Current date: {current_date}
                
                The initial search query '{search_query}' returned no results. Generate a broader, simplified search query to retrieve relevant information from the internet. Use core terms from the question, and if the question involves recent or upcoming events, include the current year (e.g., "2025"). Keep the query concise and natural, less than 10 words.
                """
                fallback_query = llm._call(fallback_query_prompt.format(question=question, current_date=current_date, search_query=search_query)).strip()
                logger.info(f"Step 7.3: Fallback search query: '{fallback_query}'")
                
                search_results = search_tool.run(fallback_query)
                logger.info(f"Step 7.4: Fallback search results: '{search_results}'")
                if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
                    logger.warning("Step 7.5: Both initial and fallback searches failed, falling back to non-time-sensitive path")
                    is_time_sensitive = False
        except ValueError as e:
            logger.warning(f"SerpAPI error: {e}, falling back to non-time-sensitive path")
            is_time_sensitive = False

        # If the search was successful, generate the answer
        if is_time_sensitive:
            logger.info("Step 8: Generate final answer from search results")
            final_answer_prompt = """
            Based on the following search results, answer the question: {question}
            Search results: {search_results}
            Current date: {current_date}
            
            Provide a direct answer, including additional relevant details about the event, such as date, location, or key participants (if applicable). Keep the answer concise and focused, limited to 3-4 sentences. If the answer is not explicitly stated, make an estimation based on the available data, or state: "Based on the provided information, I cannot determine the exact answer to '{question}'."
            """
            final_answer = llm._call(final_answer_prompt.format(question=question, search_results=search_results, current_date=current_date)).strip()
            logger.info(f"Step 8.1: Final answer (from search): '{final_answer}'")
            return final_answer

    # Step 8: If not time-sensitive (or search failed), attempt to retrieve documents
    logger.info("Step 8: Retrieve relevant documents using keyword search")
    retrieved_docs = keyword_search(vectorstore, keywords)
    if not retrieved_docs:
        logger.warning("No documents retrieved from keyword search.")
    else:
        logger.info(f"Retrieved {len(retrieved_docs)} documents: {[doc.page_content[:100] for doc in retrieved_docs]}")
        for doc in retrieved_docs:
            logger.debug(f"Retrieved document metadata: {doc.metadata}")

    # Step 9: Check document relevance
    relevant = is_relevant_docs(retrieved_docs, question) if retrieved_docs else False
    logger.info(f"Step 9: Are documents relevant to the question? {relevant}")

    # Step 10: If documents are not relevant, fall back to LLM or search
    if not retrieved_docs or not relevant:
        logger.info("Step 10: No relevant documents found, falling back to LLM or search")
        llm = TogetherLLM()
        
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
            logger.info("Step 10.1: LLM answer is insufficient, generating a search query")
            search_query_prompt = """
            Given the user question: "{question}"
            Current date: {current_date}
            
            Generate a concise, natural search query to retrieve the most relevant and up-to-date information from the internet. Use key terms from the question, and if the question involves recent or upcoming events, include the current year (e.g., "2025"). Avoid overly detailed phrasing, keeping the query under 10 words.
            """
            search_query = llm._call(search_query_prompt.format(question=question, current_date=current_date)).strip()
            logger.info(f"Generated search query: '{search_query}'")
            
            try:
                search_results = search_tool.run(search_query)
                logger.info(f"Search results: '{search_results}'")
                if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
                    logger.warning(f"SerpAPI returned no results or an error for query '{search_query}'")
                    return f"Unable to retrieve the latest information to answer the question '{question}', please try again later."
            except ValueError as e:
                logger.error(f"SerpAPI error: {e}")
                return f"Search service error, unable to answer the question '{question}', please try again later."
            
            logger.info("Step 10.2: Generate final answer from search results")
            final_answer = llm._call("""
            Based on the following search results, answer the question: {question}
            Search results: {search_results}
            Current date: {current_date}
            
            Provide only the direct answer, without any reasoning, explanation, or thought process. If the answer is not explicitly stated, make an estimation based on the available data, or state: "Based on the provided information, I cannot determine the exact answer to '{question}'."
            """.format(question=question, search_results=search_results, current_date=current_date))
            logger.info(f"Final answer (from search): '{final_answer}'")
            return final_answer
        logger.info("Step 10.1: LLM answer is sufficient, returning directly")
        return llm_answer

    # Step 11: RAFT logic
    logger.info("Step 11: Documents are relevant, proceeding with RAFT logic")
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

    llm = TogetherLLM()
    raft_response = llm._call(raft_prompt).strip()
    logger.info(f"RAFT response: '{raft_response}'")

    # Step 12: Check if RAFT response failed; if so, perform a final search
    if not raft_response or any(phrase in raft_response.lower() for phrase in no_answer_phrases):
        logger.info("Step 12: RAFT response is insufficient, performing a final search using the original question")
        final_search_query = question.strip()
        logger.info(f"Step 12.1: Final search query (using original question): '{final_search_query}'")
        
        try:
            final_search_results = search_tool.run(final_search_query)
            logger.info(f"Step 12.2: Final search results: '{final_search_results}'")
            if not final_search_results or (isinstance(final_search_results, dict) and 'error' in search_results):
                logger.warning(f"SerpAPI returned no results or an error for final search query '{final_search_query}'")
                return f"Unable to retrieve the latest information to answer the question '{question}', please try again later."
        except ValueError as e:
            logger.error(f"SerpAPI error in final search: {e}")
            return f"Search service error, unable to answer the question '{question}', please try again later."
        
        logger.info("Step 12.3: Generate final answer from search results")
        final_answer_prompt = """
        Based on the following search results, answer the question: {question}
        Search results: {search_results}
        Current date: {current_date}
        
        Provide a direct answer, including additional relevant details about the event, such as date, location, or key participants (if applicable). Keep the answer concise and focused, limited to 3-4 sentences. If the answer is not explicitly stated, make an estimation based on the available data, or state: "Based on the provided information, I cannot determine the exact answer to '{question}'."
        """
        final_answer = llm._call(final_answer_prompt.format(
            question=question,
            search_results=final_search_results,
            current_date=current_date
        )).strip()
        logger.info(f"Step 12.4: Final answer (from final search): '{final_answer}'")
        return final_answer

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