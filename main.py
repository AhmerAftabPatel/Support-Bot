from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex 
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
from typing import Optional
from llama_index.core import Response
from bs4 import BeautifulSoup
import requests

load_dotenv()

# Function to load and parse HTML content
def load_and_parse_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Preserve code blocks and JSON
        for code in soup.find_all(['pre', 'code']):
            code_text = code.get_text(strip=True)
            # Detect if it's JSON or other code
            if code_text.strip().startswith('{') or code_text.strip().startswith('['):
                code.replace_with(f"\n```json\n{code_text}\n```\n")
            else:
                code.replace_with(f"\n```\n{code_text}\n```\n")
        
        # Preserve links
        for a_tag in soup.find_all('a', href=True):
            a_tag.string = f"{a_tag.text} ({a_tag['href']})"
        
        # Preserve tables
        for table in soup.find_all('table'):
            rows_text = []
            for row in table.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                rows_text.append(" | ".join(cells))
            table_text = "\n".join(rows_text)
            table.replace_with(f"\nTABLE:\n{table_text}\n")
        
        # Preserve headers
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            header.string = f"\n{header.text}\n"
        
        # Preserve lists
        for list_tag in soup.find_all(['ul', 'ol']):
            items = [f"• {item.get_text()}" for item in list_tag.find_all('li')]
            list_tag.replace_with("\n" + "\n".join(items) + "\n")
        
        return soup.get_text(separator='\n\n', strip=True)
    else:
        print(f"Failed to fetch URL: {url}")
        return None

# Initialize models
llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 2000

pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

DATA_URL = "https://simplehtmlontent.s3.us-east-1.amazonaws.com/notion_content_1736455464998.html"
DATA_URL_2 = "https://simplehtmlontent.s3.us-east-1.amazonaws.com/notion_content_1736455489142.html"

# Check if documents are already loaded
if not os.path.exists("loaded_documents.txt"):
    print("Loading and saving documents for the first time...")
    documents_content = [load_and_parse_html(DATA_URL), load_and_parse_html(DATA_URL_2)]
    documents = [Document(text=content) for content in documents_content if content]
    
    # Save documents to txt file
    with open("loaded_documents.txt", "w", encoding="utf-8") as f:
        f.write("=== Loaded Documents ===\n\n")
        for i, doc in enumerate(documents_content, 1):
            if doc:  # Check if document was loaded successfully
                f.write(f"Document {i}:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{doc}\n")
                f.write("=" * 50 + "\n\n")
    
    print("Documents saved to 'loaded_documents.txt'")
else:
    print("Using previously loaded documents...")
    # Read the documents from file
    with open("loaded_documents.txt", "r", encoding="utf-8") as f:
        content = f.read()
        # Split the content back into documents
        documents_content = content.split("=" * 50)[:-1]  # Remove last empty split
        documents = [Document(text=doc.split("-" * 50)[1].strip()) for doc in documents_content]

print("\nDebug Document Loading:")
for i, content in enumerate(documents_content, 1):
    print(f"Document {i} status: {'Loaded' if content else 'Failed'}")
    if content:
        print(f"Document {i} length: {len(content)} characters")
    else:
        print(f"Document {i} failed to load")

# Then load and process documents
documents_content = [load_and_parse_html(DATA_URL), load_and_parse_html(DATA_URL_2)]
documents = [Document(text=content) for content in documents_content if content]

pinecode_index = pinecone_client.Index("support-bot-v2")
vector_store = PineconeVectorStore(pinecone_index=pinecode_index)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=2000, chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store
)

# pipeline.run(documents=documents)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
print(retriever)
query_engine = RetrieverQueryEngine(retriever=retriever)

# Keep all your existing chat functionality
SUPPORT_BOT_INSTRUCTIONS = """You are a customer support agent for Crustdata's APIs. You have access to the documentation through vector search.

STRICT RULES:
1. NEVER make assumptions or create examples that aren't directly from the documentation
2. If the information isn't in the provided context, respond with: "I don't have that information in the documentation"
3. Do not reference any external knowledge about the APIs
4. When providing examples, use only those shown in the documentation
5. Always cite your sources when available
6. Format all URLs as <link>URL</link> to enable proper frontend rendering

Response Format:
1. Brief explanation of the API endpoint or feature (only if required)
2. Code example (if applicable) formatted as:
   ```bash
   # For curl commands
   ```
   ```json
   # For JSON payloads
   ```
3. Important notes about (if applicable):
   • Required parameters
   • Limitations
   • Special formatting requirements
4. Reference links to documentation (if available)

Example Response:
"The search endpoint is available at <link>api.crustdata.com/screener/person/search</link>. 

Here's an example:
```bash
curl command...
```

Important Notes:
• Note 1
• Note 2

Documentation: <link>docs.example.com</link>"

Remember to follow the response format and rules strictly. Only use information from the provided documentation context."""

class SupportBotQueryEngine(BaseQueryEngine):
    def __init__(
        self, 
        base_query_engine: BaseQueryEngine, 
        system_instructions: str,
        response_mode: str = "compact"
    ):
        super().__init__(callback_manager=base_query_engine.callback_manager)
        self.base_query_engine = base_query_engine
        self.system_instructions = system_instructions
        self.response_mode = response_mode
        self.chat_history = []
        
    def _query(self, query_str: str) -> Response:
        # Combine system instructions, chat history, and current query
        formatted_query = self._format_query(query_str)
        response = self.base_query_engine.query(formatted_query)
        
        # Store the interaction in chat history
        self.chat_history.append({
            "user": query_str,
            "assistant": str(response)
        })
        
        return response
    
    def _format_query(self, query_str: str) -> str:
        # Format the complete prompt with context
        prompt = f"""System Instructions: {self.system_instructions}

Chat History:
{self._format_chat_history()}

User Query: {query_str}

Remember to follow the response format and rules strictly. Only use information from the provided documentation context."""
        
        return prompt
    
    def _format_chat_history(self) -> str:
        if not self.chat_history:
            return "No previous conversation."
            
        formatted_history = []
        for interaction in self.chat_history[-5:]:  # Keep last 5 interactions for context
            formatted_history.append(f"User: {interaction['user']}")
            formatted_history.append(f"Assistant: {interaction['assistant']}\n")
        
        return "\n".join(formatted_history)
    
    def reset_chat(self):
        """Reset the chat history"""
        self.chat_history = []

    async def _aquery(self, query_str: str) -> Response:
        # Async version of query
        return await self.base_query_engine.aquery(query_str)

    def _get_prompt_modules(self):
        # Return prompt modules from base engine
        return self.base_query_engine._get_prompt_modules()

# Initialize your query engine with the support bot wrapper
base_query_engine = RetrieverQueryEngine(
    retriever=retriever
)

support_bot = SupportBotQueryEngine(
    base_query_engine=base_query_engine,
    system_instructions=SUPPORT_BOT_INSTRUCTIONS
)

# Example usage:
def get_api_help(query: str) -> str:
    try:
        response = support_bot.query(query)
        return str(response)
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

# Interactive chat loop
def start_support_chat():
    print("Welcome to Crustdata API Support! How can I help you today? (Type 'exit' to end chat)")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for using Crustdata API Support. Goodbye!")
            break
            
        if user_input.lower() == 'reset':
            support_bot.reset_chat()
            print("\nChat history has been reset.")
            continue
            
        response = get_api_help(user_input)
        print(f"\nAssistant: {response}")

# Add this to your existing code
if __name__ == "__main__":
    start_support_chat()




