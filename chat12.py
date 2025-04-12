import streamlit as st
import together
from together import Together
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List, Mapping, Any
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
import numpy as np
from datetime import datetime
import logging
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure SerpAPI
serp_search = SerpAPIWrapper(serpapi_api_key="678e395a6c7c95e1b135322d29b35b9e7fe14712eed8900c372a31622440fbeb")
search_tool = Tool(name="Search", func=serp_search.run, description="Real-time internet search")

# Set Together AI API Key
together.api_key = "tgp_v1_BjABj4CPzcLjXO1_xh8yg0UFLoQ1cAVjpSYyozUNTNo"
st.set_page_config(page_title="RAFT Chatbot", page_icon="🤖", layout="wide")

# Custom LLM class
class TogetherLLM(LLM):
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = Together(api_key=together.api_key)
        response = client.chat.completions.create(
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

# Cache the embedding model
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector database
@st.cache_resource
def load_vector_db():
    raft_data = load_raft_dataset("raft_documents.json")
    if not raft_data:
        logger.error("No RAFT data loaded, vector database initialization failed.")
        return None

    for doc in raft_data:
        if "content" not in doc:
            logger.error(f"Missing 'content' in document {doc['doc_id']}: {doc}")
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
        logger.debug(f"Metadata for doc {meta['doc_id']}: {meta}")

    embeddings = get_embeddings_model()
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=persist_directory)
    vectorstore.persist()
    logger.info("Vector database loaded successfully.")
    return vectorstore

vector_db = load_vector_db()
if vector_db is None:
    st.error("Failed to load vector database. Please check the RAFT dataset file.")
    st.stop()

retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# RAFT question-answering logic
# def ask_raft(question, retriever):
#     messages = st.session_state.sessions.get(st.session_state.current_session, [])
#     conversation_history = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
#     current_date = datetime.now().strftime("%B %d, %Y")
    
#     # Extract the most recent assistant response (if any)
#     last_assistant_response = None
#     for message in reversed(messages):
#         if message["role"] == "assistant":
#             last_assistant_response = message["content"]
#             break
#     logger.info(f"Last Assistant Response: '{last_assistant_response}'")

#     no_answer_phrases = [
#         "i don't", "i have no information", "i'm not sure", "i cannot provide", "up-to-date", "I'm not aware",
#         "i do not have the answer", "unable to find", "no relevant information", "cutoff", "real-time",
#         "not mentioned", "does not appear", "dosn't mention", "provided context", "context provided",
#         "The new context provided does not relate to the original question", "无法提供", "我不知道", "没有相关信息",
#         "无法找到", "未提及", "不确定", "找不到相关内容", "没有", "无法", "知识截止点", "无关",
#         "not possible to determine", "absence of relevant information", "do not contain any information",
#         "no relevant content", "does not apply to the question", "exclusively discuss",
#         "i am unable to", "i lack the information", "i have no knowledge", "i am unaware",
#         "no information available", "information is missing", "cannot be determined",
#         "not found in the context", "not specified", "not available in the data",
#         "beyond my knowledge", "outside my knowledge", "not within my knowledge",
#         "no data available", "data is insufficient", "insufficient information",
#         "not covered in the documents", "not present in the documents",
#         "no record of", "no mention of", "lacking details", "details are missing",
#         "not enough context", "context is insufficient", "context does not contain",
#         "not relevant to the question", "irrelevant to the query", "unrelated to the question", "cannot determine",
#         "cannot answer due to", "unable to answer because", "answer is unavailable",
#         "information not provided", "not included in the context", "not part of the data",
#         # Chinese part
#         "无法提供", "我不知道", "没有相关信息", "无法找到", "未提及", "不确定",
#         "找不到相关内容", "没有", "无法", "知识截止点", "无关",
#         "我无法回答", "我不能回答", "我无法提供答案", "我没有答案", "我无法确定",
#         "我不能确定", "我无法给出答案", "我不能提供答案", "我无法解答",
#         "没有信息", "信息不足", "缺少信息", "没有足够的信息", "信息不全",
#         "没有相关数据", "数据不足", "数据缺失", "没有记录", "没有相关记录",
#         "信息不可用", "没有可用的信息", "没有找到信息", "信息未提供",
#         "上下文中没有", "上下文不包含", "上下文不足", "上下文没有提到",
#         "没有足够的上下文", "上下文信息不足", "上下文中未提及",
#         "文档中没有", "文档中未提到", "文档中没有相关内容",
#         "与问题无关", "与提问无关", "与问题不相关", "不相关", "无关紧要",
#         "不适用于这个问题", "与此问题无关", "与查询无关",
#         "不在我的知识范围内", "超出我的知识范围", "我的知识有限",
#         "没有这方面的知识", "我没有这方面的信息", "我对这个不了解",
#         "无法判断", "无法确认", "无法查到", "没有查到",
#         "没有提到", "没有说明", "没有详细说明", "详情缺失",
#         "无法获取", "无法得到", "无法检索到", "检索不到",
#         "没有相关资料", "资料不足", "资料中没有", "资料未包含",
#     ]
    
#     logger.info(f"Step 1: Starting RAFT process for question: '{question}'")
#     logger.info(f"Conversation History: {conversation_history}")
#     logger.info(f"Current Date: {current_date}")

#     # Step 2: Determine if the question is time-sensitive
#     # Minimal set of core time-sensitive keywords
#     time_sensitive_keywords = [
#         "recent", "latest", "current", "now", "today", "yesterday", "live",
#         "upcoming", "next", "right now", "recently", "just happened",
#         "最近", "最新", "当前", "现在", "今天", "昨天", "直播", "即将", "接下来"
#     ]

#     # Expanded regex patterns for temporal expressions
#     time_sensitive_patterns = [
#         r"last\s+(week|month|year|season|event|weekend|night|morning|day|hour|minute)",  # e.g., "last week"
#         r"this\s+(week|month|year|season|event|weekend|morning|day)",                     # e.g., "this year"
#         r"next\s+(week|month|year|season|event|weekend|day)",                            # e.g., "next event"
#         r"in\s+\d{4}",                                                                          # e.g., "in 2024"
#         r"on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",              # e.g., "on Monday"
#         r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",  # e.g., "April 2025"
#         r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",  # e.g., "April 9"
#         r"\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)",  # e.g., "9 April"
#         r"at\s+\d{1,2}:\d{2}",                                                                 # e.g., "at 14:30"
#         r"(today|yesterday|tomorrow)\s+at",                                                     # e.g., "today at"
#         r"\d{4}-\d{2}-\d{2}",                                                                  # e.g., "2025-04-09"
#     ]

#     # Check for time-sensitive keywords or patterns
#     is_time_sensitive = any(keyword in question.lower() for keyword in time_sensitive_keywords)
#     if not is_time_sensitive:
#         is_time_sensitive = any(re.search(pattern, question.lower()) for pattern in time_sensitive_patterns)

#     # If still ambiguous, use the LLM to classify
#     if not is_time_sensitive:
#         llm = TogetherLLM()
#         time_sensitive_prompt = f"""
#         Determine if the following question requires real-time or recent information to answer accurately.
#         Question: "{question}"
#         Current Date: {current_date}

#         Answer with 'Yes' if the question is time-sensitive (e.g., asks about recent events, current standings, or upcoming events).
#         Answer with 'No' if the question is not time-sensitive (e.g., asks about historical facts or general knowledge).
#         Provide only the answer ('Yes' or 'No') without any reasoning.
#         """
#         time_sensitive_result = llm._call(time_sensitive_prompt).strip().lower()
#         is_time_sensitive = time_sensitive_result == "yes"
#     logger.info(f"Step 2: Is the question time-sensitive? {is_time_sensitive}")

#     # Step 2.1: Determine if the question depends on the previous answer
#     # Minimal set of core history-dependent keywords
#     history_dependent_keywords = [
#         "previous answer", "last response", "earlier question", "just now",
#         "刚才的问题", "之前的回答", "上一个", "刚刚", "你刚说", "你提到"
#     ]

#     # Expanded regex patterns for history-dependent expressions
#     history_dependent_patterns = [
#         r"what\s+(did\s+you|was\s+the)\s+(say|mention|answer|response)",  # e.g., "What did you say?"
#         r"the\s+(previous|last|earlier)\s+(answer|response|thing)",      # e.g., "The previous answer"
#         r"(can|could)\s+you\s+(repeat|say\s+again)",                     # e.g., "Can you repeat?"
#         r"你\s*(刚|之前|上一次)\s*(说|提到|回答)",                        # e.g., "你刚说"
#         r"(repeat|restate)\s+(that|the\s+answer)",                       # e.g., "Repeat the answer"
#     ]

#     # Check for history-dependent keywords or patterns
#     is_history_dependent = any(keyword in question.lower() for keyword in history_dependent_keywords)
#     if not is_history_dependent:
#         is_history_dependent = any(re.search(pattern, question.lower()) for pattern in history_dependent_patterns)

#     # If still ambiguous, use the LLM to classify
#     if not is_history_dependent:
#         llm = TogetherLLM()
#         history_dependent_prompt = f"""
#         Determine if the following question explicitly refers to a previous answer or response in the conversation history.
#         Question: "{question}"

#         Answer with 'Yes' if the question depends on the conversation history (e.g., asks about a previous answer or what was just said).
#         Answer with 'No' if the question does not depend on the conversation history (e.g., asks a standalone question like 'What is 1+1?').
#         Provide only the answer ('Yes' or 'No') without any reasoning.
#         """
#         history_dependent_result = llm._call(history_dependent_prompt).strip().lower()
#         is_history_dependent = history_dependent_result == "yes"
#     logger.info(f"Step 2.1: Is the question history-dependent? {is_history_dependent}")

#     # Step 3: If the question is time-sensitive, prioritize using the search tool
#     if is_time_sensitive:
#         logger.info("Step 3: Time-sensitive question detected, proceeding to search")
#         llm = TogetherLLM()
        
#         # Generate search query
#         search_query_prompt = """
#         Given the user question: "{question}"
#         Current Date: {current_date}
        
#         Generate a concise and natural search query to retrieve the most relevant and up-to-date information from the internet. Use key terms from the question, ensuring the query aligns with how information is typically presented online (e.g., for event locations, include terms like 'venue' or 'held'). Include the current year (e.g., '2025') to focus on the most recent event if the question involves recent or upcoming events. Keep the query concise and clear to guarantee search accuracy.
#         """
#         search_query = llm._call(search_query_prompt.format(question=question, current_date=current_date)).strip()
#         logger.info(f"Step 4: Generated Search Query: '{search_query}'")
        
#         # Execute search
#         try:
#             search_results = search_tool.run(search_query)
#             logger.info(f"Step 5: Search Results: '{search_results}'")
#             if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
#                 logger.warning(f"SerpAPI returned no results or an error for query: '{search_query}'")
#                 fallback_query_prompt = """
#                 Given the user question: "{question}"
#                 Current Date: {current_date}
                
#                 The initial search query '{search_query}' failed to return results. Generate a broader, simplified search query to retrieve relevant information from the internet. Use the core terms from the question and include the current year (e.g., '2025') if the question involves recent or upcoming events. Keep the query concise and natural, under 10 words.
#                 """
#                 fallback_query = llm._call(fallback_query_prompt.format(question=question, current_date=current_date, search_query=search_query)).strip()
#                 logger.info(f"Step 5.1: Fallback Search Query: '{fallback_query}'")
                
#                 search_results = search_tool.run(fallback_query)
#                 logger.info(f"Step 5.2: Fallback Search Results: '{search_results}'")
#                 if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
#                     logger.warning("Step 5.3: Both initial and fallback searches failed, falling back to non-time-sensitive path")
#                     is_time_sensitive = False
#         except ValueError as e:
#             logger.warning(f"SerpAPI error: {e}, falling back to non-time-sensitive path")
#             is_time_sensitive = False

#         # If search succeeded, generate the answer
#         if is_time_sensitive:
#             logger.info("Step 6: Generating final answer from search results")
#             final_answer_prompt = """
#             Based on the following search results, answer the question: {question}
#             Search Results: {search_results}
#             Current Date: {current_date}
            
#             Provide the direct answer followed by additional relevant details about the event, such as the date, location, or key participants, if applicable. Keep the response concise, focused, and limited to 3-4 sentences. If the answer is not explicitly stated, estimate it based on available data or state: "I cannot determine the exact answer to '{question}' based on the provided information."
#             """
#             final_answer = llm._call(final_answer_prompt.format(question=question, search_results=search_results, current_date=current_date)).strip()
#             logger.info(f"Step 6.1: Final Answer (from search): '{final_answer}'")
#             return final_answer

#     # Step 4: If not time-sensitive (or search failed), attempt to retrieve documents
#     logger.info("Step 4: Retrieving relevant documents")
#     retrieved_docs = retriever.get_relevant_documents(question)
#     if not retrieved_docs:
#         logger.warning("No documents retrieved from vector database.")
#     else:
#         logger.info(f"Retrieved {len(retrieved_docs)} documents: {[doc.page_content[:100] for doc in retrieved_docs]}")
#         for doc in retrieved_docs:
#             logger.debug(f"Retrieved doc metadata: {doc.metadata}")

#     # Step 5: Check document relevance
#     def is_relevant_docs(docs, question):
#         embeddings = get_embeddings_model()
#         question_embedding = embeddings.embed_query(question)
#         for doc in docs:
#             doc_embedding = embeddings.embed_query(doc.page_content)
#             similarity = np.dot(question_embedding, doc_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(doc_embedding))
#             logger.debug(f"Similarity with doc '{doc.page_content[:50]}...': {similarity}")
#             if similarity > 0.5:
#                 return True
#         return False

#     relevant = is_relevant_docs(retrieved_docs, question) if retrieved_docs else False
#     logger.info(f"Step 5: Documents relevant to question? {relevant}")

#     # Step 6: If documents are not relevant, fall back to LLM or search
#     if not retrieved_docs or not relevant:
#         logger.info("Step 6: No relevant docs found, falling back to LLM or search")
#         llm = TogetherLLM()
        
#         # If the question is history-dependent, use a specific prompt
#         if is_history_dependent:
#             llm_prompt = """
#             Conversation History: {conversation_history}
#             Last Assistant Response: {last_assistant_response}
#             User Question: {question}
#             Current Date: {current_date}

#             The question refers to a previous answer or last response. Use the Last Assistant Response provided above to answer the question. If the Last Assistant Response is 'None' or does not contain the necessary information to answer the question, return: "I cannot determine the previous answer to answer '{question}' because the conversation history is insufficient."
#             Provide only the direct answer without any reasoning, explanations, or thought process.
#             """
#             llm_prompt = llm_prompt.format(
#                 conversation_history=conversation_history,
#                 last_assistant_response=last_assistant_response if last_assistant_response else 'None',
#                 question=question,
#                 current_date=current_date
#             )
#         else:
#             # For non-history-dependent questions, answer directly without checking history
#             llm_prompt = """
#             User Question: {question}
#             Current Date: {current_date}

#             Answer the question directly using your general knowledge. Provide only the direct answer without any reasoning, explanations, or thought process.
#             """
#             llm_prompt = llm_prompt.format(question=question, current_date=current_date)

#         llm_answer = llm._call(llm_prompt).strip()
#         logger.info(f"LLM Answer (no relevant docs): '{llm_answer}'")
        
#         if not llm_answer or any(phrase in llm_answer.lower() for phrase in no_answer_phrases):
#             logger.info("Step 6.1: LLM answer insufficient, generating search query")
#             search_query_prompt = """
#             Given the user question: "{question}"
#             Current Date: {current_date}
            
#             Generate a concise and natural search query to retrieve the most relevant and up-to-date information from the internet. Use key terms from the question and include the current year (e.g., '2025') if the question involves recent or upcoming events. Avoid overly detailed phrasing and keep the query under 10 words.
#             """
#             search_query = llm._call(search_query_prompt.format(question=question, current_date=current_date)).strip()
#             logger.info(f"Generated Search Query: '{search_query}'")
            
#             try:
#                 search_results = search_tool.run(search_query)
#                 logger.info(f"Search Results: '{search_results}'")
#                 if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
#                     logger.warning(f"SerpAPI returned no results or an error for query: '{search_query}'")
#                     return f"Unable to retrieve the latest information to answer the question '{question}', please try again later."
#             except ValueError as e:
#                 logger.error(f"SerpAPI error: {e}")
#                 return f"Search service error, unable to answer the question '{question}', please try again later."
            
#             logger.info("Step 6.2: Generating final answer from search results")
#             final_answer = llm._call("""
#             Based on the following search results, answer the question: {question}
#             Search Results: {search_results}
#             Current Date: {current_date}
            
#             Provide only the direct answer without any reasoning, explanations, or thought process. If the answer is not explicitly stated, estimate it based on available data or state: "I cannot determine the exact answer to '{question}' based on the provided information."
#             """.format(question=question, search_results=search_results, current_date=current_date))
#             logger.info(f"Final Answer (from search): '{final_answer}'")
#             return final_answer
#         logger.info("Step 6.1: LLM answer sufficient, returning directly")
#         return llm_answer

#     # Step 7: RAFT logic
#     logger.info("Step 7: Documents relevant, proceeding with RAFT logic")
#     mid_point = len(retrieved_docs) // 2
#     golden_docs = retrieved_docs[:mid_point]
#     distractor_docs = retrieved_docs[mid_point:]
    
#     def format_doc_content(doc):
#         metadata = doc.metadata
#         title = metadata.get('title', 'Unknown Title')
#         content = metadata.get('content', 'Content not available')
#         return f"Title: {title}\nContent: {content}"

#     golden_docs_content = [format_doc_content(doc) for doc in golden_docs]
#     distractor_docs_content = [format_doc_content(doc) for doc in distractor_docs]
    
#     # If the question is history-dependent, include the last assistant response in the RAFT prompt
#     if is_history_dependent:
#         raft_prompt = """
#         You are a model trained with RAFT (Retrieval Augmented Fine-Tuning), capable of extracting answers from provided documents while ignoring irrelevant information.
#         The question refers to a previous answer or last response. Use the Last Assistant Response provided below to answer the question. If the Last Assistant Response is 'None' or does not contain the necessary information, use the "Golden Documents" to find the answer, ignoring the "Distractor Documents". If the documents and history lack sufficient information, use your general knowledge based on the current date ({current_date}) to provide the most accurate answer possible. If you cannot determine the answer, return: "I cannot determine the previous answer to answer '{question}' because the conversation history is insufficient."

#         User Question: {question}
#         Conversation History: {conversation_history}
#         Last Assistant Response: {last_assistant_response}

#         Golden Documents:
#         {golden_docs_content}

#         Distractor Documents:
#         {distractor_docs_content}

#         Provide only the direct answer without any reasoning, explanations, or thought process.
#         """
#         raft_prompt = raft_prompt.format(
#             current_date=current_date,
#             question=question,
#             conversation_history=conversation_history,
#             last_assistant_response=last_assistant_response if last_assistant_response else 'None',
#             golden_docs_content='\n'.join(golden_docs_content),
#             distractor_docs_content='\n'.join(distractor_docs_content)
#         )
#     else:
#         raft_prompt = """
#         You are a model trained with RAFT (Retrieval Augmented Fine-Tuning), capable of extracting answers from provided documents while ignoring irrelevant information.
#         Use the "Golden Documents" to find the answer to the user's question, ignoring the "Distractor Documents". If the documents lack sufficient information, use your general knowledge based on the current date ({current_date}) to provide the most accurate answer possible. If you cannot determine the answer, explicitly state: "I don’t have enough information to determine the answer to '{question}' as of {current_date}."

#         User Question: {question}

#         Golden Documents:
#         {golden_docs_content}

#         Distractor Documents:
#         {distractor_docs_content}

#         Provide only the direct answer without any reasoning, explanations, or thought process.
#         """
#         raft_prompt = raft_prompt.format(
#             current_date=current_date,
#             question=question,
#             golden_docs_content='\n'.join(golden_docs_content),
#             distractor_docs_content='\n'.join(distractor_docs_content)
#         )

#     llm = TogetherLLM()
#     raft_response = llm._call(raft_prompt).strip()
#     logger.info(f"RAFT Response: '{raft_response}'")

#     # Step 8: Check if RAFT response failed; if so, perform a final search
#     if not raft_response or any(phrase in raft_response.lower() for phrase in no_answer_phrases):
#         logger.info("Step 8: RAFT response insufficient, performing final search with original question")
#         final_search_query = question.strip()
#         logger.info(f"Step 8.1: Final Search Query (using original question): '{final_search_query}'")
        
#         try:
#             final_search_results = search_tool.run(final_search_query)
#             logger.info(f"Step 8.2: Final Search Results: '{final_search_results}'")
#             if not final_search_results or (isinstance(final_search_results, dict) and 'error' in final_search_results):
#                 logger.warning(f"SerpAPI returned no results or an error for final search query: '{final_search_query}'")
#                 return f"Unable to retrieve the latest information to answer the question '{question}', please try again later."
#         except ValueError as e:
#             logger.error(f"SerpAPI error in final search: {e}")
#             return f"Search service error, unable to answer the question '{question}', please try again later."
        
#         logger.info("Step 8.3: Generating final answer from search results")
#         final_answer_prompt = """
#         Based on the following search results, answer the question: {question}
#         Search Results: {search_results}
#         Current Date: {current_date}
        
#         Provide the direct answer followed by additional relevant details about the event, such as the date, location, or key participants, if applicable. Keep the response concise, focused, and limited to 3-4 sentences. If the answer is not explicitly stated, estimate it based on available data or state: "I cannot determine the exact answer to '{question}' based on the provided information."
#         """
#         final_answer = llm._call(final_answer_prompt.format(
#             question=question,
#             search_results=final_search_results,
#             current_date=current_date
#         )).strip()
#         logger.info(f"Step 8.4: Final Answer (from final search): '{final_answer}'")
#         return final_answer

#     return raft_response

# RAFT question-answering logic
def ask_raft(question, retriever):
    messages = st.session_state.sessions.get(st.session_state.current_session, [])
    conversation_history = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Extract the most recent assistant response (if any)
    last_assistant_response = None
    for message in reversed(messages):
        if message["role"] == "assistant":
            last_assistant_response = message["content"]
            break
    logger.info(f"Last Assistant Response: '{last_assistant_response}'")

    no_answer_phrases = [
        "i don't", "i have no information", "i'm not sure", "i cannot provide", "up-to-date", "I'm not aware",
        "i do not have the answer", "unable to find", "no relevant information", "cutoff", "real-time",
        "not mentioned", "does not appear", "dosn't mention", "provided context", "context provided",
        "The new context provided does not relate to the original question", "无法提供", "我不知道", "没有相关信息",
        "无法找到", "未提及", "不确定", "找不到相关内容", "没有", "无法", "知识截止点", "无关",
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
        "cannot answer due to", "unable to answer because", "answer is unavailable",
        "information not provided", "not included in the context", "not part of the data",
        # Chinese part
        "无法提供", "我不知道", "没有相关信息", "无法找到", "未提及", "不确定",
        "找不到相关内容", "没有", "无法", "知识截止点", "无关",
        "我无法回答", "我不能回答", "我无法提供答案", "我没有答案", "我无法确定",
        "我不能确定", "我无法给出答案", "我不能提供答案", "我无法解答",
        "没有信息", "信息不足", "缺少信息", "没有足够的信息", "信息不全",
        "没有相关数据", "数据不足", "数据缺失", "没有记录", "没有相关记录",
        "信息不可用", "没有可用的信息", "没有找到信息", "信息未提供",
        "上下文中没有", "上下文不包含", "上下文不足", "上下文没有提到",
        "没有足够的上下文", "上下文信息不足", "上下文中未提及",
        "文档中没有", "文档中未提到", "文档中没有相关内容",
        "与问题无关", "与提问无关", "与问题不相关", "不相关", "无关紧要",
        "不适用于这个问题", "与此问题无关", "与查询无关",
        "不在我的知识范围内", "超出我的知识范围", "我的知识有限",
        "没有这方面的知识", "我没有这方面的信息", "我对这个不了解",
        "无法判断", "无法确认", "无法查到", "没有查到",
        "没有提到", "没有说明", "没有详细说明", "详情缺失",
        "无法获取", "无法得到", "无法检索到", "检索不到",
        "没有相关资料", "资料不足", "资料中没有", "资料未包含",
    ]
    
    logger.info(f"Step 1: Starting RAFT process for question: '{question}'")
    logger.info(f"Conversation History: {conversation_history}")
    logger.info(f"Current Date: {current_date}")

    # New Step 2: Initial document search
    logger.info("Step 2: Performing initial document search")
    initial_docs = retriever.get_relevant_documents(question)
    if not initial_docs:
        logger.warning("No documents retrieved from vector database for initial search.")
    else:
        logger.info(f"Retrieved {len(initial_docs)} documents: {[doc.page_content[:100] for doc in initial_docs]}")
        for doc in initial_docs:
            logger.debug(f"Initial retrieved doc metadata: {doc.metadata}")

    # Check document relevance
    def is_relevant_docs(docs, question):
        embeddings = get_embeddings_model()
        question_embedding = embeddings.embed_query(question)
        for doc in docs:
            doc_embedding = embeddings.embed_query(doc.page_content)
            similarity = np.dot(question_embedding, doc_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(doc_embedding))
            logger.debug(f"Similarity with doc '{doc.page_content[:50]}...': {similarity}")
            if similarity > 0.5:
                return True
        return False

    initial_relevant = is_relevant_docs(initial_docs, question) if initial_docs else False
    logger.info(f"Step 2.1: Initial documents relevant to question? {initial_relevant}")

    # If relevant documents are found, try to extract the answer
    if initial_relevant:
        logger.info("Step 2.2: Relevant documents found, attempting to extract answer")
        def format_doc_content(doc):
            metadata = doc.metadata
            title = metadata.get('title', 'Unknown Title')
            content = metadata.get('content', 'Content not available')
            return f"Title: {title}\nContent: {content}"

        initial_docs_content = [format_doc_content(doc) for doc in initial_docs]
        llm = TogetherLLM()
        initial_answer_prompt = """
        Use the following documents to answer the user's question. Provide only the direct answer without any reasoning, explanations, or thought process. If you cannot determine the answer, explicitly state: "I don’t have enough information to determine the answer to '{question}' as of {current_date}."

        User Question: {question}

        Documents:
        {docs_content}

        Provide only the direct answer.
        """
        initial_answer = llm._call(initial_answer_prompt.format(
            question=question,
            current_date=current_date,
            docs_content='\n'.join(initial_docs_content)
        )).strip()
        logger.info(f"Step 2.3: Initial Answer from documents: '{initial_answer}'")

        # If the initial answer is sufficient, return it
        if initial_answer and not any(phrase in initial_answer.lower() for phrase in no_answer_phrases):
            logger.info("Step 2.4: Initial answer sufficient, returning directly")
            return initial_answer
        else:
            logger.info("Step 2.4: Initial answer insufficient, proceeding to next steps")

    # Step 3: Determine if the question is time-sensitive
    # Minimal set of core time-sensitive keywords
    time_sensitive_keywords = [
        "recent", "latest", "current", "now", "today", "yesterday", "live",
        "upcoming", "next", "right now", "recently", "just happened",
        "最近", "最新", "当前", "现在", "今天", "昨天", "直播", "即将", "接下来"
    ]

    # Expanded regex patterns for temporal expressions
    time_sensitive_patterns = [
        r"last\s+(week|month|year|season|event|weekend|night|morning|day|hour|minute)",  # e.g., "last week"
        r"this\s+(week|month|year|season|event|weekend|morning|day)",                     # e.g., "this year"
        r"next\s+(week|month|year|season|event|weekend|day)",                            # e.g., "next event"
        r"in\s+\d{4}",                                                                          # e.g., "in 2024"
        r"on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",              # e.g., "on Monday"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",  # e.g., "April 2025"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",  # e.g., "April 9"
        r"\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)",  # e.g., "9 April"
        r"at\s+\d{1,2}:\d{2}",                                                                 # e.g., "at 14:30"
        r"(today|yesterday|tomorrow)\s+at",                                                     # e.g., "today at"
        r"\d{4}-\d{2}-\d{2}",                                                                  # e.g., "2025-04-09"
    ]

    # Check for time-sensitive keywords or patterns
    is_time_sensitive = any(keyword in question.lower() for keyword in time_sensitive_keywords)
    if not is_time_sensitive:
        is_time_sensitive = any(re.search(pattern, question.lower()) for pattern in time_sensitive_patterns)

    # If still ambiguous, use the LLM to classify
    if not is_time_sensitive:
        llm = TogetherLLM()
        time_sensitive_prompt = f"""
        Determine if the following question requires real-time or recent information to answer accurately.
        Question: "{question}"
        Current Date: {current_date}

        Answer with 'Yes' if the question is time-sensitive (e.g., asks about recent events, current standings, or upcoming events).
        Answer with 'No' if the question is not time-sensitive (e.g., asks about historical facts or general knowledge).
        Provide only the answer ('Yes' or 'No') without any reasoning.
        """
        time_sensitive_result = llm._call(time_sensitive_prompt).strip().lower()
        is_time_sensitive = time_sensitive_result == "yes"
    logger.info(f"Step 3: Is the question time-sensitive? {is_time_sensitive}")

    # Step 4: Determine if the question depends on the previous answer
    # Minimal set of core history-dependent keywords
    history_dependent_keywords = [
        "previous answer", "last response", "earlier question", "just now",
        "刚才的问题", "之前的回答", "上一个", "刚刚", "你刚说", "你提到"
    ]

    # Expanded regex patterns for history-dependent expressions
    history_dependent_patterns = [
        r"what\s+(did\s+you|was\s+the)\s+(say|mention|answer|response)",  # e.g., "What did you say?"
        r"the\s+(previous|last|earlier)\s+(answer|response|thing)",      # e.g., "The previous answer"
        r"(can|could)\s+you\s+(repeat|say\s+again)",                     # e.g., "Can you repeat?"
        r"你\s*(刚|之前|上一次)\s*(说|提到|回答)",                        # e.g., "你刚说"
        r"(repeat|restate)\s+(that|the\s+answer)",                       # e.g., "Repeat the answer"
    ]

    # Check for history-dependent keywords or patterns
    is_history_dependent = any(keyword in question.lower() for keyword in history_dependent_keywords)
    if not is_history_dependent:
        is_history_dependent = any(re.search(pattern, question.lower()) for pattern in history_dependent_patterns)

    # If still ambiguous, use the LLM to classify
    if not is_history_dependent:
        llm = TogetherLLM()
        history_dependent_prompt = f"""
        Determine if the following question explicitly refers to a previous answer or response in the conversation history.
        Question: "{question}"

        Answer with 'Yes' if the question depends on the conversation history (e.g., asks about a previous answer or what was just said).
        Answer with 'No' if the question does not depend on the conversation history (e.g., asks a standalone question like 'What is 1+1?').
        Provide only the answer ('Yes' or 'No') without any reasoning.
        """
        history_dependent_result = llm._call(history_dependent_prompt).strip().lower()
        is_history_dependent = history_dependent_result == "yes"
    logger.info(f"Step 4: Is the question history-dependent? {is_history_dependent}")

    # Step 5: If the question is time-sensitive, prioritize using the search tool
    if is_time_sensitive:
        logger.info("Step 5: Time-sensitive question detected, proceeding to search")
        llm = TogetherLLM()
        
        # Generate search query
        search_query_prompt = """
        Given the user question: "{question}"
        Current Date: {current_date}
        
        Generate a concise and natural search query to retrieve the most relevant and up-to-date information from the internet. Use key terms from the question, ensuring the query aligns with how information is typically presented online (e.g., for event locations, include terms like 'venue' or 'held'). Include the current year (e.g., '2025') to focus on the most recent event if the question involves recent or upcoming events. Keep the query concise and clear to guarantee search accuracy.
        """
        search_query = llm._call(search_query_prompt.format(question=question, current_date=current_date)).strip()
        logger.info(f"Step 6: Generated Search Query: '{search_query}'")
        
        # Execute search
        try:
            search_results = search_tool.run(search_query)
            logger.info(f"Step 7: Search Results: '{search_results}'")
            if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
                logger.warning(f"SerpAPI returned no results or an error for query: '{search_query}'")
                fallback_query_prompt = """
                Given the user question: "{question}"
                Current Date: {current_date}
                
                The initial search query '{search_query}' failed to return results. Generate a broader, simplified search query to retrieve relevant information from the internet. Use the core terms from the question and include the current year (e.g., '2025') if the question involves recent or upcoming events. Keep the query concise and natural, under 10 words.
                """
                fallback_query = llm._call(fallback_query_prompt.format(question=question, current_date=current_date, search_query=search_query)).strip()
                logger.info(f"Step 7.1: Fallback Search Query: '{fallback_query}'")
                
                search_results = search_tool.run(fallback_query)
                logger.info(f"Step 7.2: Fallback Search Results: '{search_results}'")
                if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
                    logger.warning("Step 7.3: Both initial and fallback searches failed, falling back to non-time-sensitive path")
                    is_time_sensitive = False
        except ValueError as e:
            logger.warning(f"SerpAPI error: {e}, falling back to non-time-sensitive path")
            is_time_sensitive = False

        # If search succeeded, generate the answer
        if is_time_sensitive:
            logger.info("Step 8: Generating final answer from search results")
            final_answer_prompt = """
            Based on the following search results, answer the question: {question}
            Search Results: {search_results}
            Current Date: {current_date}
            
            Provide the direct answer followed by additional relevant details about the event, such as the date, location, or key participants, if applicable. Keep the response concise, focused, and limited to 3-4 sentences. If the answer is not explicitly stated, estimate it based on available data or state: "I cannot determine the exact answer to '{question}' based on the provided information."
            """
            final_answer = llm._call(final_answer_prompt.format(question=question, search_results=search_results, current_date=current_date)).strip()
            logger.info(f"Step 8.1: Final Answer (from search): '{final_answer}'")
            return final_answer

    # Step 6: If not time-sensitive (or search failed), attempt to retrieve documents
    logger.info("Step 6: Retrieving relevant documents")
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        logger.warning("No documents retrieved from vector database.")
    else:
        logger.info(f"Retrieved {len(retrieved_docs)} documents: {[doc.page_content[:100] for doc in retrieved_docs]}")
        for doc in retrieved_docs:
            logger.debug(f"Retrieved doc metadata: {doc.metadata}")

    # Step 7: Check document relevance
    relevant = is_relevant_docs(retrieved_docs, question) if retrieved_docs else False
    logger.info(f"Step 7: Documents relevant to question? {relevant}")

    # Step 8: If documents are not relevant, fall back to LLM or search
    if not retrieved_docs or not relevant:
        logger.info("Step 8: No relevant docs found, falling back to LLM or search")
        llm = TogetherLLM()
        
        # If the question is history-dependent, use a specific prompt
        if is_history_dependent:
            llm_prompt = """
            Conversation History: {conversation_history}
            Last Assistant Response: {last_assistant_response}
            User Question: {question}
            Current Date: {current_date}

            The question refers to a previous answer or last response. Use the Last Assistant Response provided above to answer the question. If the Last Assistant Response is 'None' or does not contain the necessary information to answer the question, return: "I cannot determine the previous answer to answer '{question}' because the conversation history is insufficient."
            Provide only the direct answer without any reasoning, explanations, or thought process.
            """
            llm_prompt = llm_prompt.format(
                conversation_history=conversation_history,
                last_assistant_response=last_assistant_response if last_assistant_response else 'None',
                question=question,
                current_date=current_date
            )
        else:
            # For non-history-dependent questions, answer directly without checking history
            llm_prompt = """
            User Question: {question}
            Current Date: {current_date}

            Answer the question directly using your general knowledge. Provide only the direct answer without any reasoning, explanations, or thought process.
            """
            llm_prompt = llm_prompt.format(question=question, current_date=current_date)

        llm_answer = llm._call(llm_prompt).strip()
        logger.info(f"LLM Answer (no relevant docs): '{llm_answer}'")
        
        if not llm_answer or any(phrase in llm_answer.lower() for phrase in no_answer_phrases):
            logger.info("Step 8.1: LLM answer insufficient, generating search query")
            search_query_prompt = """
            Given the user question: "{question}"
            Current Date: {current_date}
            
            Generate a concise and natural search query to retrieve the most relevant and up-to-date information from the internet. Use key terms from the question and include the current year (e.g., '2025') if the question involves recent or upcoming events. Avoid overly detailed phrasing and keep the query under 10 words.
            """
            search_query = llm._call(search_query_prompt.format(question=question, current_date=current_date)).strip()
            logger.info(f"Generated Search Query: '{search_query}'")
            
            try:
                search_results = search_tool.run(search_query)
                logger.info(f"Search Results: '{search_results}'")
                if not search_results or (isinstance(search_results, dict) and 'error' in search_results):
                    logger.warning(f"SerpAPI returned no results or an error for query: '{search_query}'")
                    return f"Unable to retrieve the latest information to answer the question '{question}', please try again later."
            except ValueError as e:
                logger.error(f"SerpAPI error: {e}")
                return f"Search service error, unable to answer the question '{question}', please try again later."
            
            logger.info("Step 8.2: Generating final answer from search results")
            final_answer = llm._call("""
            Based on the following search results, answer the question: {question}
            Search Results: {search_results}
            Current Date: {current_date}
            
            Provide only the direct answer without any reasoning, explanations, or thought process. If the answer is not explicitly stated, estimate it based on available data or state: "I cannot determine the exact answer to '{question}' based on the provided information."
            """.format(question=question, search_results=search_results, current_date=current_date))
            logger.info(f"Final Answer (from search): '{final_answer}'")
            return final_answer
        logger.info("Step 8.1: LLM answer sufficient, returning directly")
        return llm_answer

    # Step 9: RAFT logic
    logger.info("Step 9: Documents relevant, proceeding with RAFT logic")
    mid_point = len(retrieved_docs) // 2
    golden_docs = retrieved_docs[:mid_point]
    distractor_docs = retrieved_docs[mid_point:]
    
    def format_doc_content(doc):
        metadata = doc.metadata
        title = metadata.get('title', 'Unknown Title')
        content = metadata.get('content', 'Content not available')
        return f"Title: {title}\nContent: {content}"

    golden_docs_content = [format_doc_content(doc) for doc in golden_docs]
    distractor_docs_content = [format_doc_content(doc) for doc in distractor_docs]
    
    # If the question is history-dependent, include the last assistant response in the RAFT prompt
    if is_history_dependent:
        raft_prompt = """
        You are a model trained with RAFT (Retrieval Augmented Fine-Tuning), capable of extracting answers from provided documents while ignoring irrelevant information.
        The question refers to a previous answer or last response. Use the Last Assistant Response provided below to answer the question. If the Last Assistant Response is 'None' or does not contain the necessary information, use the "Golden Documents" to find the answer, ignoring the "Distractor Documents". If the documents and history lack sufficient information, use your general knowledge based on the current date ({current_date}) to provide the most accurate answer possible. If you cannot determine the answer, return: "I cannot determine the previous answer to answer '{question}' because the conversation history is insufficient."

        User Question: {question}
        Conversation History: {conversation_history}
        Last Assistant Response: {last_assistant_response}

        Golden Documents:
        {golden_docs_content}

        Distractor Documents:
        {distractor_docs_content}

        Provide only the direct answer without any reasoning, explanations, or thought process.
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
        You are a model trained with RAFT (Retrieval Augmented Fine-Tuning), capable of extracting answers from provided documents while ignoring irrelevant information.
        Use the "Golden Documents" to find the answer to the user's question, ignoring the "Distractor Documents". If the documents lack sufficient information, use your general knowledge based on the current date ({current_date}) to provide the most accurate answer possible. If you cannot determine the answer, explicitly state: "I don’t have enough information to determine the answer to '{question}' as of {current_date}."

        User Question: {question}

        Golden Documents:
        {golden_docs_content}

        Distractor Documents:
        {distractor_docs_content}

        Provide only the direct answer without any reasoning, explanations, or thought process.
        """
        raft_prompt = raft_prompt.format(
            current_date=current_date,
            question=question,
            golden_docs_content='\n'.join(golden_docs_content),
            distractor_docs_content='\n'.join(distractor_docs_content)
        )

    llm = TogetherLLM()
    raft_response = llm._call(raft_prompt).strip()
    logger.info(f"RAFT Response: '{raft_response}'")

    # Step 10: Check if RAFT response failed; if so, perform a final search
    if not raft_response or any(phrase in raft_response.lower() for phrase in no_answer_phrases):
        logger.info("Step 10: RAFT response insufficient, performing final search with original question")
        final_search_query = question.strip()
        logger.info(f"Step 10.1: Final Search Query (using original question): '{final_search_query}'")
        
        try:
            final_search_results = search_tool.run(final_search_query)
            logger.info(f"Step 10.2: Final Search Results: '{final_search_results}'")
            if not final_search_results or (isinstance(final_search_results, dict) and 'error' in final_search_results):
                logger.warning(f"SerpAPI returned no results or an error for final search query: '{final_search_query}'")
                return f"Unable to retrieve the latest information to answer the question '{question}', please try again later."
        except ValueError as e:
            logger.error(f"SerpAPI error in final search: {e}")
            return f"Search service error, unable to answer the question '{question}', please try again later."
        
        logger.info("Step 10.3: Generating final answer from search results")
        final_answer_prompt = """
        Based on the following search results, answer the question: {question}
        Search Results: {search_results}
        Current Date: {current_date}
        
        Provide the direct answer followed by additional relevant details about the event, such as the date, location, or key participants, if applicable. Keep the response concise, focused, and limited to 3-4 sentences. If the answer is not explicitly stated, estimate it based on available data or state: "I cannot determine the exact answer to '{question}' based on the provided information."
        """
        final_answer = llm._call(final_answer_prompt.format(
            question=question,
            search_results=final_search_results,
            current_date=current_date
        )).strip()
        logger.info(f"Step 10.4: Final Answer (from final search): '{final_answer}'")
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

user_message = st.text_input("💬 Enter your question:", value="", key="user_message_input")
if st.button("Send"):
    if user_message:
        messages.append({"role": "user", "content": user_message})
        raft_response = ask_raft(user_message, retriever)
        messages.append({"role": "assistant", "content": raft_response})
        st.session_state.sessions[st.session_state.current_session] = messages
        st.session_state.user_message = ""
        st.rerun()