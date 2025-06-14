---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480

# Agent DiNeo 🤖

A sophisticated multi-modal AI agent built with LangGraph that combines intelligent workflow orchestration, vector search capabilities, and comprehensive tool integration for complex problem-solving tasks.

## 🏆 Achievement

Built as part of the HuggingFace AI Agents course - **80% completion score with Certificate of Excellence**

## ✨ Features

### Core Capabilities
- **Multi-Modal Processing**: Handle text, images, files, and YouTube videos
- **Semantic Search**: Advanced RAG pipeline with vector similarity matching
- **Web Integration**: Real-time web search using Serper API
- **Academic Research**: Wikipedia and arXiv search capabilities
- **File Analysis**: Excel, CSV, and document processing
- **Image Processing**: OCR and visual content analysis using Gemini Vision
- **Mathematical Operations**: Built-in calculator functions

### Intelligent Workflow
- **LangGraph Architecture**: State-based agent workflow management
- **Tool Orchestration**: Seamless integration of multiple specialized tools
- **Context Awareness**: Maintains conversation context across interactions
- **Error Handling**: Robust error management and recovery

## 🛠 Technical Architecture

### Core Components
- **Agent Engine**: LangGraph-powered workflow management
- **Vector Store**: Supabase-based semantic search with custom embedding handling
- **LLM Integration**: Support for Groq, Google Gemini, and HuggingFace models
- **Tool Ecosystem**: Modular tool architecture with rate limiting and debugging

### Key Technologies
- **LangGraph**: Advanced AI agent workflows
- **Supabase**: Vector database with pgvector extension
- **HuggingFace**: Sentence transformers for embeddings
- **Gradio**: Web interface for user interaction
- **Google Gemini**: Multi-modal AI capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required API keys (see Environment Setup)

### Installation
```bash
git clone https://github.com/yourusername/Agent_DiNeo.git
cd Agent_DiNeo
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with:
```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
SERPER_API_KEY=your_serper_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Usage

#### Local Development
```bash
python agent.py
```

#### Web Interface
```bash
python app.py
```

#### HuggingFace Spaces
The agent is deployed and ready to use at: [Agent DiNeo Space](https://huggingface.co/spaces/digvijayingole55/Agent_DiNeo)

## 📁 Project Structure

```
Agent_DiNeo/
├── agent.py           # Main agent implementation with LangGraph workflow
├── app.py            # Gradio web interface and evaluation runner
├── tools.py          # Comprehensive tool implementations
├── system_prompt.txt # Agent system instructions
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## 🔧 Key Components

### Agent Workflow (`agent.py`)
- **Retriever Node**: Semantic search in vector database
- **Assistant Node**: LLM-powered response generation
- **Tool Node**: Dynamic tool execution
- **State Management**: Conversation context preservation

### Tool Ecosystem (`tools.py`)
- **Web Search**: Serper API integration with rate limiting
- **Wikipedia Search**: Academic and general knowledge queries
- **arXiv Search**: Scientific paper retrieval
- **File Processing**: Excel, CSV, and document analysis
- **Image Analysis**: OCR and visual content extraction
- **Video Analysis**: YouTube content processing

### Vector Store Integration
- **Custom Embedding Handler**: Solves JSON string embedding storage issues
- **Cosine Similarity**: Client-side similarity calculations
- **Supabase Integration**: Scalable vector database backend

## 🎯 Advanced Features

### Semantic Search
The agent implements a sophisticated RAG pipeline that:
- Converts queries to embeddings using HuggingFace transformers
- Performs semantic similarity search in Supabase vector store
- Handles edge cases like JSON string embeddings
- Returns contextually relevant documents

### Multi-Modal Processing
- **Text Analysis**: Natural language processing and understanding
- **Image Processing**: OCR, visual analysis, and content extraction
- **File Analysis**: Structured data processing (Excel, CSV)
- **Video Analysis**: YouTube content extraction and analysis

### Rate Limiting & Optimization
- **Global Rate Limiter**: Coordinates API calls across tools
- **Exponential Backoff**: Handles rate limiting gracefully
- **Caching**: Reduces redundant API calls
- **Error Recovery**: Robust handling of API failures

## 🔍 Troubleshooting

### Common Issues

**Vector Store Connection**
- Ensure `SUPABASE_URL` and `SUPABASE_KEY` are set
- Verify Supabase project has pgvector extension enabled

**API Rate Limits**
- The agent implements automatic rate limiting
- If you encounter persistent rate limits, check your API quotas

**Embedding Issues**
- The agent handles JSON string embeddings automatically
- For new vector stores, ensure embeddings are stored as proper vectors

**Model Loading**
- HuggingFace models download automatically on first use
- Ensure stable internet connection for model downloads

## 🌐 Live Demo

Experience the agent in action: [Agent DiNeo on HuggingFace Spaces](https://huggingface.co/spaces/digvijayingole55/Agent_DiNeo)

## 📊 Performance

- **Evaluation Score**: 80% on HuggingFace AI Agents course assessment
- **Multi-Modal Support**: Handles text, images, files, and videos
- **Response Time**: Optimized for real-time interaction
- **Scalability**: Deployed on HuggingFace Spaces for public access

## 🛡 Security Features

- **Environment Variable Protection**: Sensitive data stored securely
- **Rate Limiting**: Prevents API abuse
- **Error Sanitization**: Prevents information leakage in error messages
- **Input Validation**: Robust input checking and sanitization
