"# Support-Bot" 
# Crustdata API Support Bot

A conversational AI system leveraging advanced vector similarity search and large language models to provide contextual API documentation assistance.

## Documentation Sources

### Notion Documentation Integration
The system ingests and processes documentation from two primary Notion sources:

1. **Discovery & Enrichment API Documentation**
   - Source: `https://crustdata.notion.site/Crustdata-Discovery-And-Enrichment-API-c66d5236e8ea40df8af114f6d447ab48`
   - Content Type: API Documentation
   - Topics Covered:
     - Authentication methods
     - Endpoint specifications
     - Request/Response formats
     - Rate limiting details

2. **Dataset API Examples**
   - Source: `https://crustdata.notion.site/Crustdata-Dataset-API-Detailed-Examples-b83bd0f1ec09452bb0c2cac811bba88c`
   - Content Type: Implementation Examples
   - Topics Covered:
     - Code samples
     - Use case scenarios
     - Integration patterns
     - Best practices

## Technical Architecture

### Frontend (Next.js 15.1.4)
- React-based SPA with client-side state management
- Tailwind CSS for atomic styling and responsive design
- WebSocket-ready architecture for real-time communication
- Custom message formatting engine for handling code blocks and API references

### Backend (Python)
- LlamaIndex for semantic document processing and retrieval
- Google's Gemini LLM for natural language understanding
- Pinecone vector database for high-dimensional similarity search
- Custom ingestion pipeline with BeautifulSoup for HTML parsing

## Core Technologies

### Vector Search Engine
- Pinecone vector store implementation
- Embedding dimension: 768 (Gemini embedding-001 model)
- Cosine similarity metric for nearest neighbor search
- Top-k retrieval with k=5 for context aggregation

### Document Processing
- Custom HTML parsing with preservation of:
  - Code blocks (JSON/non-JSON detection)
  - Hyperlinks
  - Tabular data
  - Headers (h1-h4)
  - Lists (ordered/unordered)
- Sentence-level chunking with 2000-token windows and 20-token overlap

### LLM Integration
- Gemini API with custom prompt engineering
- Context-aware response generation
- Strict adherence to documentation boundaries
- Format-preserving response templates

## System Components

### Vector Store Integration

### Query Engine
Custom implementation with:
- Chat history management
- System instruction injection
- Async query support
- Response formatting

### Frontend State Management

## Performance Considerations

### Vector Search
- Optimized chunk size (2000 tokens) for context retention
- Minimal chunk overlap (20 tokens) to reduce index size
- Efficient vector similarity computation using Pinecone's distributed architecture

### Response Generation
- Streaming-ready architecture
- Efficient message formatting with regex-based parsing
- Optimized DOM updates for chat interface

## Development Setup

### Prerequisites
- Node.js ≥ 18.0.0
- Python ≥ 3.8
- Pinecone API access
- Google API key (Gemini access)

### Environment Configuration
Required environment variables:

GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_key
NEXT_PUBLIC_API_URL=backend_url

### Installation

Backend:

bash
pip install llama-index pinecone-client google.generativeai beautifulsoup4

Frontend:

bash
npm install
npm run dev

## Technical Limitations

- Maximum context window: 2000 tokens per chunk
- Rate limits: Dependent on Gemini API tier
- Vector dimension: Fixed at 768 (Gemini embedding model)
- Response time: Variable based on vector search complexity

## Future Optimizations

- Implement vector store caching layer
- Add streaming response support
- Optimize chunk size based on content type
- Implement rate limiting and request queuing
- Add support for multi-modal content processing

## Contributing

Refer to the component documentation for specific implementation details.