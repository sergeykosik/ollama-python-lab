from typing import Dict, List
import ollama
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Model to use with Ollama
model_name = "llama3.2:latest"  # Update with the model you have available in Ollama

# List of documents to process
documents = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-servers-discord/",
    "https://beebom.com/how-list-groups-linux/",
    "https://beebom.com/how-open-port-linux/",
    "https://beebom.com/linux-vs-windows/",
]

# Initialize Ollama client
ollama_client = ollama.Client(host='http://host.docker.internal:11434')  # Update with your Ollama host if needed

# Custom embedding function using Ollama
class OllamaEmbeddings:
    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.client = ollama_client
    
    def embed_documents(self, texts):
        """Generate embeddings for a list of documents"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings
    
    def embed_query(self, text):
        """Generate embeddings for a query"""
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]


def scrape_docs(urls: List[str]) -> List[Dict]:
    """Scrape content from URLs using requests and BeautifulSoup"""
    try:
        documents = []
        
        for url in urls:
            try:
                # Use requests to get the page content
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Raise an error for bad status codes
                
                # Parse the HTML content with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup.select('header, footer, nav, .ads, .sidebar, .comments, script, style'):
                    element.decompose()
                
                # Extract the text content
                text = soup.get_text(separator='\n', strip=True)
                
                # Create a Document object
                doc = Document(page_content=text, metadata={"source": url})
                documents.append(doc)
                
                print(f"\nSuccessfully loaded: {url}")
                print(f"Content length: {len(text)} characters")
                
            except Exception as e:
                print(f"Error loading {url}: {str(e)}")
        
        print(f"\nSuccessfully loaded {len(documents)} documents")
        return documents

    except Exception as e:
        print(f"Error during document loading: {str(e)}")
        return []


def create_vector_store(texts: List[str], metadatas: List[Dict]):
    """Create vector store using ChromaDB with Ollama embeddings"""
    # Check if we have documents to process
    if not texts:
        print("No texts to create embeddings for. Returning None.")
        return None
    
    embeddings = OllamaEmbeddings()
    db = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings)
    return db


def setup_qa_chain(db):
    """Set up QA chain with Ollama model"""
    retriever = db.as_retriever()

    # Create a custom prompt template
    prompt = ChatPromptTemplate.from_template(
        """
    Please provide a polite and helpful response to the following question, utilizing the provided context. 
    Ensure that the tone remains professional, courteous, and empathetic, and tailor your response to directly address the inquiry. 

### Context:
{context}

### Question: 
{question}

### Polite Response:
In your response, consider including:
- Acknowledge the user's query and express gratitude for the opportunity to assist.
- Provide a clear and concise answer that directly addresses the question.
- Use positive language and maintain a supportive tone throughout.
- If applicable, include relevant information or resources that could help further.
- Conclude by inviting any follow-up questions or providing encouragement for the user's pursuit of information."""
    )

    # Create a custom LLM class for Ollama
    class OllamaLLM:
        def __init__(self, model_name=model_name):
            self.model_name = model_name
            self.client = ollama_client
        
        def invoke(self, prompt):
            """Generate a response from Ollama"""
            response = self.client.chat(model=self.model_name, messages=[
                {"role": "user", "content": prompt}
            ])
            return response["message"]["content"]

    # Initialize our Ollama LLM
    llm = OllamaLLM()

    # Create the chain
    def generate_response(input_dict):
        # Format the prompt with context and question
        formatted_prompt = prompt.format(
            context="\n".join([doc.page_content for doc in input_dict["context"]]),
            question=input_dict["question"]
        )
        
        # Get response from LLM
        return llm.invoke(formatted_prompt)
    
    # Define the chain function
    def chain_invoke(question):
        # Get context from retriever
        context = retriever.invoke(question)
        # Generate response
        response = generate_response({"context": context, "question": question})
        return response

    return chain_invoke, retriever


def split_documents(pages_content: List[Dict]) -> tuple:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_texts, all_metadatas = [], []
    for document in pages_content:
        # Extract text from Document object
        text = document.page_content
        source = document.metadata.get("source", "")

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": source})

    print(f"Created {len(all_texts)} chunks of text")
    return all_texts, all_metadatas


def process_query(chain_and_retriever, query: str):
    """Process a query and return response"""
    try:
        chain, retriever = chain_and_retriever  # Unpack the tuple

        # Get the response using our chain
        response = chain(query)

        # Get the sources using the retriever
        docs = retriever.invoke(query)
        sources_str = ", ".join([doc.metadata.get("source", "") for doc in docs])

        return {"answer": response, "sources": sources_str}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            "answer": "I apologize, but I encountered an error while processing your question.",
            "sources": "",
        }


def main():
    # 1. Scrape documents
    print("Scraping documents...")
    pages_content = scrape_docs(documents)

    # 2. Split documents
    print("Splitting documents...")
    all_texts, all_metadatas = split_documents(pages_content)

    # 3. Create vector store
    print("Creating vector store...")
    db = create_vector_store(all_texts, all_metadatas)
    
    # Exit if no database was created
    if db is None:
        print("No vector database created. Exiting.")
        return

    # 4. Set up QA chain
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(db)

    # 5. Interactive query loop
    print("\nReady for questions! (Type 'quit' to exit)")
    while True:
        query = input("\nEnter your question: ").strip()

        if not query:
            continue

        if query.lower() == "quit":
            break

        result = process_query(qa_chain, query)

        print("\nResponse:")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"].split(","):
                print("- " + source.strip())


if __name__ == "__main__":
    main()