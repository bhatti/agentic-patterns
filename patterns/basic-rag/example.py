"""
Basic RAG (Retrieval-Augmented Generation) Pattern

This example demonstrates the Basic RAG pattern with two pipelines:

INDEXING PIPELINE (Preparatory):
    Step 1: Load documents from knowledge sources
    Step 2: Chunk documents into manageable pieces
    Step 3: Store chunks in searchable index

RETRIEVAL-GENERATION PIPELINE (Runtime):
    Step 1: Receive user query
    Step 2: Retrieve relevant chunks from index
    Step 3: Ground prompt with retrieved chunks
    Step 4: Generate response using LLM

USE CASE: Product Documentation RAG
    Realistic scenario: Building a RAG system to answer questions
    about product features, APIs, and usage from documentation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    id: str
    title: str
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    id: str
    text: str
    document_id: str
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# INDEXING PIPELINE
# ============================================================================

class DocumentLoader:
    """
    Step 1: Load documents from knowledge sources.
    
    In production, this would load from files, databases, APIs, etc.
    """
    
    def load_documents(self, source: str) -> List[Document]:
        """
        Load documents from a source.
        
        Args:
            source: Source identifier (file path, database, API, etc.)
            
        Returns:
            List of documents
        """
        # Simulate loading product documentation
        # In production, this would read from actual files/databases
        
        documents = [
            Document(
                id="doc-001",
                title="API Authentication Guide",
                content="""
                API Authentication Overview
                
                Our API uses OAuth 2.0 for authentication. To get started:
                
                1. Register your application to get client credentials
                2. Request an access token using the /oauth/token endpoint
                3. Include the access token in the Authorization header
                
                Example request:
                POST /oauth/token
                {
                    "grant_type": "client_credentials",
                    "client_id": "your_client_id",
                    "client_secret": "your_client_secret"
                }
                
                The access token expires after 3600 seconds. Refresh tokens
                can be used to obtain new access tokens without re-authenticating.
                """,
                source="docs/api/authentication.md",
                metadata={"category": "API", "version": "v2.0"}
            ),
            Document(
                id="doc-002",
                title="User Management API",
                content="""
                User Management API Reference
                
                The User Management API allows you to create, read, update, and
                delete user accounts in the system.
                
                Endpoints:
                - GET /api/v1/users - List all users
                - GET /api/v1/users/{id} - Get user by ID
                - POST /api/v1/users - Create new user
                - PUT /api/v1/users/{id} - Update user
                - DELETE /api/v1/users/{id} - Delete user
                
                All endpoints require authentication via Bearer token in the
                Authorization header.
                
                Rate Limits: 100 requests per minute per API key.
                """,
                source="docs/api/users.md",
                metadata={"category": "API", "version": "v2.0"}
            ),
            Document(
                id="doc-003",
                title="Getting Started Guide",
                content="""
                Getting Started with Our Platform
                
                Welcome! This guide will help you get started quickly.
                
                Step 1: Sign up for an account
                Visit our website and create an account. You'll receive a
                confirmation email to verify your email address.
                
                Step 2: Generate API Keys
                Navigate to Settings > API Keys and generate a new key pair.
                Store these securely - you won't be able to see the secret again.
                
                Step 3: Make Your First API Call
                Use the authentication guide to get an access token, then
                make a simple GET request to /api/v1/users to list users.
                
                Need help? Check our documentation or contact support.
                """,
                source="docs/getting-started.md",
                metadata={"category": "Guide", "version": "v1.0"}
            ),
        ]
        
        logger.info(f"Loaded {len(documents)} documents from {source}")
        return documents


class TextSplitter:
    """
    Step 2: Chunk documents into manageable pieces.
    
    Splits documents into chunks that preserve context while being
    small enough for efficient retrieval and LLM processing.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk (characters)
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of chunks
        """
        text = document.content.strip()
        chunks = []
        
        # Simple splitting by sentences (production would use better NLP)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                metadata = {"title": document.title, "source": document.source}
                if document.metadata:
                    metadata.update(document.metadata)
                
                chunks.append(Chunk(
                    id=f"{document.id}-chunk-{chunk_index}",
                    text=chunk_text,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata=metadata
                ))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self._overlap_sentence_count():]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = {"title": document.title, "source": document.source}
            if document.metadata:
                metadata.update(document.metadata)
            
            chunks.append(Chunk(
                id=f"{document.id}-chunk-{chunk_index}",
                text=chunk_text,
                document_id=document.id,
                chunk_index=chunk_index,
                metadata=metadata
            ))
        
        logger.info(f"Split document {document.id} into {len(chunks)} chunks")
        return chunks
    
    def _overlap_sentence_count(self) -> int:
        """Calculate how many sentences to include in overlap."""
        # Simple heuristic: overlap ~10% of chunk
        return max(1, int(self.chunk_overlap / 100))


class Index:
    """
    Step 3: Store chunks in searchable index.
    
    In production, this would use a vector database, search engine,
    or specialized RAG framework. For basic RAG, we use simple keyword search.
    """
    
    def __init__(self):
        """Initialize the index."""
        self.chunks: List[Chunk] = []
        self.chunk_by_id: Dict[str, Chunk] = {}
        self.document_chunks: Dict[str, List[Chunk]] = {}
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks to the index.
        
        Args:
            chunks: List of chunks to add
        """
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_by_id[chunk.id] = chunk
            
            if chunk.document_id not in self.document_chunks:
                self.document_chunks[chunk.document_id] = []
            self.document_chunks[chunk.document_id].append(chunk)
        
        logger.info(f"Added {len(chunks)} chunks to index. Total: {len(self.chunks)}")
    
    def get_all_chunks(self) -> List[Chunk]:
        """Get all chunks in the index."""
        return self.chunks


# ============================================================================
# RETRIEVAL-GENERATION PIPELINE
# ============================================================================

class Retriever:
    """
    Step 1-2: Retrieve relevant chunks from index.
    
    Searches the index for chunks relevant to the query.
    Basic implementation uses keyword matching.
    """
    
    def __init__(self, index: Index, top_k: int = 3):
        """
        Initialize retriever.
        
        Args:
            index: The knowledge index
            top_k: Number of top chunks to retrieve
        """
        self.index = index
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question/query
            
        Returns:
            List of relevant chunks, sorted by relevance
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Simple keyword-based scoring
        scored_chunks = []
        
        for chunk in self.index.get_all_chunks():
            chunk_lower = chunk.text.lower()
            chunk_words = set(chunk_lower.split())
            
            # Calculate relevance score (simple word overlap)
            common_words = query_words.intersection(chunk_words)
            score = len(common_words) / max(len(query_words), 1)
            
            # Boost score if query words appear multiple times
            for word in query_words:
                score += chunk_lower.count(word) * 0.1
            
            scored_chunks.append((score, chunk))
        
        # Sort by score (descending) and return top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in scored_chunks[:self.top_k]]
        
        logger.info(f"Retrieved {len(top_chunks)} chunks for query: {query[:50]}...")
        return top_chunks


class RAGGenerator:
    """
    Step 3-4: Ground prompt with retrieved chunks and generate response.
    
    Constructs a prompt with retrieved context and generates response using LLM.
    """
    
    def __init__(self, retriever: Retriever):
        """
        Initialize RAG generator.
        
        Args:
            retriever: Retriever for finding relevant chunks
        """
        self.retriever = retriever
    
    def generate(self, query: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Generate response using RAG.
        
        Args:
            query: User's question
            use_llm: Whether to use actual LLM (False for demo)
            
        Returns:
            Response with answer and sources
        """
        # Step 1-2: Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve(query)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "chunks_used": 0
            }
        
        # Step 3: Ground prompt with retrieved chunks
        context = self._build_context(relevant_chunks)
        prompt = self._build_prompt(query, context)
        
        # Step 4: Generate response
        if use_llm:
            # In production, this would call an actual LLM
            # from shared.ollama_client import OllamaClient
            # client = OllamaClient()
            # answer = client.generate(prompt, model="llama3")
            answer = "[LLM would generate answer here using retrieved context]"
        else:
            # Simulate answer generation
            answer = self._simulate_answer(query, relevant_chunks)
        
        return {
            "answer": answer,
            "sources": [self._format_source(chunk) for chunk in relevant_chunks],
            "chunks_used": len(relevant_chunks),
            "context": context[:200] + "..." if len(context) > 200 else context
        }
    
    def _build_context(self, chunks: List[Chunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Document {i}: {chunk.metadata.get('title', 'Unknown')}]\n{chunk.text}")
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt with context and query."""
        return f"""Use the following documentation to answer the question. If the answer cannot be found in the documentation, say so.

Documentation:
{context}

Question: {query}

Answer:"""
    
    def _simulate_answer(self, query: str, chunks: List[Chunk]) -> str:
        """Simulate answer generation for demonstration."""
        # Simple simulation: extract relevant information
        combined_text = " ".join([chunk.text for chunk in chunks])
        
        # Try to find direct answer in chunks
        query_lower = query.lower()
        if "authentication" in query_lower or "token" in query_lower:
            if "OAuth" in combined_text:
                return "Our API uses OAuth 2.0 for authentication. You need to register your application to get client credentials, then request an access token using the /oauth/token endpoint. Include the access token in the Authorization header for subsequent requests."
        
        if "endpoint" in query_lower or "api" in query_lower:
            return "Based on the documentation, the API provides various endpoints for user management, authentication, and other operations. All endpoints require authentication via Bearer token in the Authorization header."
        
        return f"Based on the documentation: {combined_text[:200]}..."
    
    def _format_source(self, chunk: Chunk) -> Dict[str, str]:
        """Format chunk as source reference."""
        return {
            "title": chunk.metadata.get("title", "Unknown"),
            "source": chunk.metadata.get("source", "Unknown"),
            "chunk_id": chunk.id
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_indexing_pipeline():
    """Demonstrate the indexing pipeline."""
    print("\n" + "="*70)
    print("INDEXING PIPELINE (Preparatory)")
    print("="*70)
    
    print("\n📚 Step 1: Load Documents")
    loader = DocumentLoader()
    documents = loader.load_documents("product_docs")
    
    for doc in documents:
        print(f"   • {doc.title} ({len(doc.content)} chars)")
    
    print("\n✂️  Step 2: Chunk Documents")
    splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_document(doc)
        all_chunks.extend(chunks)
        print(f"   • {doc.title}: {len(chunks)} chunks")
    
    print(f"\n   Total chunks created: {len(all_chunks)}")
    
    print("\n💾 Step 3: Store in Index")
    index = Index()
    index.add_chunks(all_chunks)
    
    print(f"   Index contains {len(index.get_all_chunks())} chunks")
    
    print("\n✅ Indexing complete - knowledge base ready for queries")
    return index


def demonstrate_retrieval_generation(index: Index):
    """Demonstrate the retrieval-generation pipeline."""
    print("\n" + "="*70)
    print("RETRIEVAL-GENERATION PIPELINE (Runtime)")
    print("="*70)
    
    retriever = Retriever(index, top_k=3)
    generator = RAGGenerator(retriever)
    
    queries = [
        "How do I authenticate with the API?",
        "What endpoints are available for user management?",
        "How do I get started with the platform?"
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        result = generator.generate(query)
        
        print(f"\n📝 Answer:")
        print(f"   {result['answer']}")
        
        print(f"\n📚 Sources ({result['chunks_used']} chunks used):")
        for source in result['sources']:
            print(f"   • {source['title']} ({source['source']})")
        
        print(f"\n🔍 Context Preview:")
        print(f"   {result['context'][:150]}...")


def show_rag_architecture():
    """Show RAG architecture diagram."""
    print("\n" + "="*70)
    print("🏗️  RAG ARCHITECTURE")
    print("="*70)
    
    print("""
    INDEXING PIPELINE (One-time/Periodic):
    
    Knowledge Sources
         ↓
    [Document Loader]
         ↓
    [Text Splitter] → Chunks
         ↓
    [Index/Store] → Knowledge Base
    
    
    RETRIEVAL-GENERATION PIPELINE (Runtime):
    
    User Query
         ↓
    [Retriever] → Relevant Chunks
         ↓
    [Prompt Builder] → Grounded Prompt
         ↓
    [LLM] → Response
    """)


def show_real_world_example():
    """Show realistic use case example."""
    print("\n" + "="*70)
    print("🌍 REAL-WORLD USE CASE: Product Documentation RAG")
    print("="*70)
    
    print("\nScenario: Build a Q&A system for product documentation")
    print("Challenge: Documentation is large, constantly updated, and contains")
    print("specific technical details that base LLMs don't know")
    
    print("\n❌ Without RAG:")
    print("   • LLM doesn't know about your specific API")
    print("   • May provide outdated or incorrect information")
    print("   • Can't access private documentation")
    print("   • May hallucinate API endpoints or features")
    
    print("\n✅ With RAG:")
    print("   • Retrieves relevant documentation chunks")
    print("   • Grounds LLM response with actual docs")
    print("   • Provides accurate, up-to-date information")
    print("   • Can cite sources for transparency")
    
    print("\nExample Flow:")
    print("   1. User: 'How do I authenticate?'")
    print("   2. Retriever finds: Authentication guide chunks")
    print("   3. LLM generates answer using retrieved chunks")
    print("   4. Response includes source references")
    
    print("\n" + "="*70)
    print("🎯 Impact: Accurate answers, up-to-date info, source transparency")
    print("="*70)


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 BASIC RAG PATTERN")
    print("="*70)
    
    print("\n📋 Pattern Overview:")
    print("   Two-pipeline approach: Indexing (preparatory) and")
    print("   Retrieval-Generation (runtime)")
    print("   Augments LLM with external knowledge sources")
    
    # Demonstrate indexing pipeline
    index = demonstrate_indexing_pipeline()
    
    # Show architecture
    show_rag_architecture()
    
    # Demonstrate retrieval-generation pipeline
    demonstrate_retrieval_generation(index)
    
    # Show use case
    show_real_world_example()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for detailed explanation")
    print("   2. Choose appropriate chunking strategy")
    print("   3. Select retrieval method (keyword, semantic, hybrid)")
    print("   4. Integrate with LLM for generation")
    print("   5. Add source attribution")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

