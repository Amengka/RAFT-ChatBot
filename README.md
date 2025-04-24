# RAFT Chatbot

## Overview
RAFT Chatbot is a Python-based conversational AI application designed to answer user queries using a retrieval-augmented approach. It leverages keyword-based document retrieval to fetch relevant information from a preloaded dataset and integrates real-time internet search for time-sensitive queries. Built with Streamlit for the frontend, LangChain for memory management, and Together AI's LLM for natural language processing, this chatbot provides a seamless and interactive user experience.

## Features
- **Keyword-Based Retrieval**: Retrieves documents from a dataset by matching extracted keywords, limited to the top 10 most relevant results.
- **Time-Sensitive Search**: Detects time-sensitive queries and uses SerpAPI to fetch real-time information from the web.
- **Conversation Memory**: Maintains conversation history across sessions using LangChain's `ConversationBufferMemory`.
- **Interactive UI**: Provides a user-friendly interface via Streamlit, with support for multiple chat sessions.
- **Robust Logging**: Includes detailed logging for debugging and monitoring retrieval and answering processes.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/raft-chatbot.git
   cd raft-chatbot
   ```
2. **Install Dependencies**:
   Ensure you have Python 3.12 installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include `streamlit`, `together`, `langchain`, `langchain-community`, `sentence-transformers`, `numpy`, and `serpapi`.
3. **Set API Keys**:
   - Obtain a Together AI API key and set it in the code (`together.api_key`).
   - Obtain a SerpAPI key and set it in the `initialize_chatbot` function.
4. **Prepare the Dataset**:
   Place your dataset in `raft_documents.json` format in the project root directory.

## Usage
1. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run chatbot-modified.py
   ```
   To run the application using docker, run: 
   ```
   docker-compose up
   ```
2. **Interact with the Chatbot**:
   - Open the app in your browser (default: `http://localhost:8501`).
   - Enter your question in the input box and click "Send".
   - View the conversation history and switch between sessions using the sidebar.


## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.