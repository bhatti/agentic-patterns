"""
Index-Aware Retrieval Pattern - Real-World Problem Solver

PROBLEM: Technical API Documentation Q&A System
    Users ask questions in natural language ("How do I log in?"), but your
    API documentation uses technical terminology ("OAuth 2.0 authentication",
    "access token", "client credentials"). Basic RAG fails because:
    - Vocabulary mismatch: "log in" ≠ "authentication"
    - Fine details: Answer hidden in large chunks
    - Holistic answers: Requires connecting multiple concepts

SOLUTION: Index-Aware Retrieval with Multiple Techniques
    - Hypothetical Document Embedding (HyDE): Generate hypothetical answer,
      then match chunks to that answer
    - Query Expansion: Translate user terms to technical terms
    - Hybrid Search: Combine keyword (BM25) and semantic search
    - GraphRAG: Retrieve related chunks via graph structure

This example implements a working API documentation Q&A system that handles
vocabulary mismatches and finds answers requiring multiple concepts.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import math
import re

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of documentation."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    document_id: str = ""
    chunk_index: int = 0
    related_chunk_ids: List[str] = field(default_factory=list)


# ============================================================================
# TECHNIQUE 1: HYPOTHETICAL DOCUMENT EMBEDDING (HyDE)
# ============================================================================

class HyDEGenerator:
    """
    Hypothetical Document Embedding (HyDE) Generator.
    
    Generates a hypothetical answer to the query, then uses that answer
    to find relevant chunks. This bridges vocabulary gaps.
    """
    
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate hypothetical answer to query.
        
        In production, this would use an LLM:
        prompt = f"Answer this question: {query}"
        return llm.generate(prompt)
        """
        # Simulate LLM generating hypothetical answer
        # This answer uses terminology that matches the knowledge base
        
        query_lower = query.lower()
        
        if "log" in query_lower or "login" in query_lower or "sign in" in query_lower:
            return "To authenticate and access the API, you need to obtain an OAuth 2.0 access token using client credentials. First, register your application to get client_id and client_secret. Then make a POST request to /oauth/token endpoint with grant_type=client_credentials."
        
        elif "token" in query_lower or "auth" in query_lower:
            return "Authentication requires OAuth 2.0 access tokens. Use client credentials flow: POST to /oauth/token with client_id, client_secret, and grant_type=client_credentials. The response contains an access_token to use in Authorization header."
        
        elif "error" in query_lower or "problem" in query_lower:
            return "Common issues include invalid credentials, expired tokens, or incorrect endpoint URLs. Check error codes: 401 for authentication failures, 403 for insufficient permissions, 404 for invalid endpoints."
        
        else:
            return f"To use the API, you need to authenticate using OAuth 2.0, obtain an access token, and include it in the Authorization header of your requests."
    
    def retrieve_with_hyde(self, query: str, chunks: List[DocumentChunk], 
                          top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve chunks using HyDE approach.
        
        1. Generate hypothetical answer
        2. Embed hypothetical answer
        3. Find chunks similar to hypothetical answer
        """
        # Step 1: Generate hypothetical answer
        hypothetical_answer = self.generate_hypothetical_answer(query)
        
        # Step 2: Embed hypothetical answer
        hyde_embedding = self.embedding_generator.generate_embedding(hypothetical_answer)
        
        # Step 3: Find chunks similar to hypothetical answer
        scored_chunks = []
        for chunk in chunks:
            if chunk.embedding:
                similarity = self.embedding_generator.cosine_similarity(
                    hyde_embedding, chunk.embedding
                )
                scored_chunks.append((chunk, similarity))
        
        # Sort by similarity
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_k]


# ============================================================================
# TECHNIQUE 2: QUERY EXPANSION
# ============================================================================

class QueryExpander:
    """
    Query Expansion: Add context and translate terms.
    
    Expands user queries with technical terms and synonyms to match
    the vocabulary used in the knowledge base.
    """
    
    def __init__(self):
        # Term translation dictionary: user term -> technical terms
        self.term_translations = {
            "log in": ["authentication", "oauth", "access token", "login"],
            "login": ["authentication", "oauth", "access token"],
            "sign in": ["authentication", "oauth", "access token"],
            "authenticate": ["oauth", "access token", "client credentials"],
            "token": ["access token", "oauth token", "bearer token"],
            "error": ["error code", "status code", "exception", "failure"],
            "problem": ["error", "issue", "troubleshooting", "debug"],
            "api": ["endpoint", "rest api", "http", "request"],
            "call": ["request", "http request", "api call", "endpoint"],
            "data": ["payload", "request body", "json", "response"]
        }
        
        # Context additions for common queries
        self.context_additions = {
            "log": ["authentication", "security", "credentials"],
            "token": ["oauth", "authorization", "bearer"],
            "error": ["troubleshooting", "status code", "debugging"]
        }
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with translations and context.
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query with technical terms
        """
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Add term translations
        for user_term, tech_terms in self.term_translations.items():
            if user_term in query_lower:
                expanded_terms.extend(tech_terms)
        
        # Add context based on query content
        for key, context_terms in self.context_additions.items():
            if key in query_lower:
                expanded_terms.extend(context_terms)
        
        # Remove duplicates and join
        expanded_query = " ".join(list(dict.fromkeys(expanded_terms)))
        
        return expanded_query
    
    def generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple query variations."""
        variations = [query]
        
        # Add expanded version
        variations.append(self.expand_query(query))
        
        # Add variations with different phrasings
        if "how" in query.lower():
            variations.append(query.replace("how", "what"))
            variations.append(query.replace("how", "way to"))
        
        return variations


# ============================================================================
# TECHNIQUE 3: HYBRID SEARCH (BM25 + Semantic)
# ============================================================================

class BM25Scorer:
    """BM25 keyword-based scoring."""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.avg_doc_length = self._calculate_avg_length()
        self.k1 = 1.5  # BM25 parameter
        self.b = 0.75  # BM25 parameter
    
    def _calculate_avg_length(self) -> float:
        """Calculate average document length."""
        if not self.chunks:
            return 1.0
        total_length = sum(len(chunk.content.split()) for chunk in self.chunks)
        return total_length / len(self.chunks)
    
    def score(self, query: str, chunk: DocumentChunk) -> float:
        """Calculate BM25 score for query-chunk pair."""
        query_terms = query.lower().split()
        chunk_terms = chunk.content.lower().split()
        doc_length = len(chunk_terms)
        
        score = 0.0
        term_freqs = {}
        for term in chunk_terms:
            term_freqs[term] = term_freqs.get(term, 0) + 1
        
        for term in query_terms:
            if term in term_freqs:
                tf = term_freqs[term]
                idf = math.log((len(self.chunks) + 1) / (1 + 1))  # Simplified IDF
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score += idf * (numerator / denominator)
        
        return score


class HybridRetriever:
    """
    Hybrid Search: Combines BM25 (keyword) and Semantic (embedding) search.
    
    Final score = α × BM25_score + (1-α) × semantic_score
    """
    
    def __init__(self, chunks: List[DocumentChunk], embedding_generator, alpha: float = 0.4):
        """
        Initialize hybrid retriever.
        
        Args:
            chunks: List of document chunks
            embedding_generator: Generator for embeddings
            alpha: Weight for BM25 (0.4 = 40% BM25, 60% semantic)
        """
        self.chunks = chunks
        self.embedding_generator = embedding_generator
        self.alpha = alpha
        self.bm25_scorer = BM25Scorer(chunks)
        
        # Generate embeddings for all chunks
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = embedding_generator.generate_embedding(chunk.content)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve chunks using hybrid search.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Calculate scores
        scored_chunks = []
        
        for chunk in self.chunks:
            # BM25 score (keyword-based)
            bm25_score = self.bm25_scorer.score(query, chunk)
            
            # Normalize BM25 score (0-1 range)
            bm25_normalized = min(bm25_score / 10.0, 1.0) if bm25_score > 0 else 0.0
            
            # Semantic score (embedding-based)
            if chunk.embedding:
                semantic_score = self.embedding_generator.cosine_similarity(
                    query_embedding, chunk.embedding
                )
            else:
                semantic_score = 0.0
            
            # Hybrid score: weighted combination
            hybrid_score = self.alpha * bm25_normalized + (1 - self.alpha) * semantic_score
            
            scored_chunks.append((chunk, hybrid_score))
        
        # Sort by hybrid score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_k]


# ============================================================================
# TECHNIQUE 4: GRAPHRAG
# ============================================================================

class GraphRAG:
    """
    GraphRAG: Retrieve related chunks via graph structure.
    
    Stores chunks as nodes in a graph, creates edges between related chunks,
    then traverses graph to find related information.
    """
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Dict[str, List[str]]:
        """
        Build graph structure from chunks.
        
        Creates edges between chunks that:
        - Share keywords
        - Are in same document
        - Reference each other
        """
        graph = {chunk_id: [] for chunk_id in self.chunks.keys()}
        
        for chunk_id, chunk in self.chunks.items():
            # Connect chunks in same document
            for other_id, other_chunk in self.chunks.items():
                if chunk_id != other_id:
                    # Same document
                    if chunk.document_id == other_chunk.document_id:
                        graph[chunk_id].append(other_id)
                    
                    # Share keywords
                    if chunk.keywords and other_chunk.keywords:
                        common_keywords = set(chunk.keywords) & set(other_chunk.keywords)
                        if common_keywords:
                            graph[chunk_id].append(other_id)
                    
                    # Content references (simplified)
                    if chunk.id in other_chunk.content or other_chunk.id in chunk.content:
                        graph[chunk_id].append(other_id)
        
        return graph
    
    def retrieve_related(self, initial_chunk_id: str, depth: int = 1, 
                        max_related: int = 3) -> List[DocumentChunk]:
        """
        Retrieve related chunks via graph traversal.
        
        Args:
            initial_chunk_id: ID of initial relevant chunk
            depth: How many hops to traverse
            max_related: Maximum related chunks to return
            
        Returns:
            List of related chunks
        """
        if initial_chunk_id not in self.graph:
            return []
        
        related_ids = set()
        current_level = [initial_chunk_id]
        
        for _ in range(depth):
            next_level = []
            for chunk_id in current_level:
                if chunk_id in self.graph:
                    related_ids.update(self.graph[chunk_id])
                    next_level.extend(self.graph[chunk_id])
            current_level = next_level
        
        # Remove initial chunk
        related_ids.discard(initial_chunk_id)
        
        # Get related chunks
        related_chunks = [self.chunks[cid] for cid in list(related_ids)[:max_related] 
                         if cid in self.chunks]
        
        return related_chunks


# ============================================================================
# API DOCUMENTATION Q&A SYSTEM
# ============================================================================

class APIDocumentationQA:
    """
    API Documentation Q&A System with Index-Aware Retrieval.
    
    This solves the real problem: answering user questions in natural language
    when documentation uses technical terminology.
    """
    
    def __init__(self, chunks: List[DocumentChunk], embedding_generator):
        self.chunks = chunks
        self.embedding_generator = embedding_generator
        
        # Initialize all techniques
        self.hyde_generator = HyDEGenerator(embedding_generator)
        self.query_expander = QueryExpander()
        self.hybrid_retriever = HybridRetriever(chunks, embedding_generator)
        self.graphrag = GraphRAG(chunks)
    
    def query(self, question: str, use_hyde: bool = True, 
             use_expansion: bool = True, use_hybrid: bool = True,
             use_graphrag: bool = True) -> Dict[str, Any]:
        """
        Query the documentation using index-aware retrieval.
        
        This solves the real problem by combining multiple retrieval techniques.
        """
        all_chunks = []
        techniques_used = []
        
        # Technique 1: HyDE
        if use_hyde:
            hyde_results = self.hyde_generator.retrieve_with_hyde(question, self.chunks, top_k=3)
            all_chunks.extend([chunk for chunk, _ in hyde_results])
            techniques_used.append("HyDE")
        
        # Technique 2: Query Expansion + Hybrid Search
        if use_expansion and use_hybrid:
            expanded_query = self.query_expander.expand_query(question)
            hybrid_results = self.hybrid_retriever.retrieve(expanded_query, top_k=3)
            all_chunks.extend([chunk for chunk, _ in hybrid_results])
            techniques_used.append("Query Expansion + Hybrid Search")
        
        # Technique 3: GraphRAG (find related chunks)
        if use_graphrag and all_chunks:
            # Get initial chunk
            initial_chunk = all_chunks[0]
            related_chunks = self.graphrag.retrieve_related(initial_chunk.id, depth=1, max_related=2)
            all_chunks.extend(related_chunks)
            techniques_used.append("GraphRAG")
        
        # Remove duplicates
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)
        
        # Generate answer (simulated)
        answer = self._generate_answer(unique_chunks, question)
        
        return {
            "answer": answer,
            "chunks_used": len(unique_chunks),
            "techniques": techniques_used,
            "sources": [
                {
                    "content": chunk.content[:150] + "...",
                    "document": chunk.document_id
                }
                for chunk in unique_chunks[:3]
            ]
        }
    
    def _generate_answer(self, chunks: List[DocumentChunk], question: str) -> str:
        """Generate answer from chunks (simulated)."""
        if not chunks:
            return "No relevant information found."
        
        # Use first chunk as primary answer
        primary = chunks[0]
        answer = f"Based on the documentation: {primary.content[:200]}..."
        
        if len(chunks) > 1:
            answer += f"\n\nAdditional context from related sections: {chunks[1].content[:100]}..."
        
        return answer


# ============================================================================
# DEMONSTRATION
# ============================================================================

def create_sample_documentation() -> List[DocumentChunk]:
    """Create sample API documentation chunks."""
    chunks = [
        DocumentChunk(
            id="chunk-1",
            content="OAuth 2.0 Authentication: To authenticate with the API, you must first register your application to obtain client credentials (client_id and client_secret). Then make a POST request to the /oauth/token endpoint with grant_type=client_credentials.",
            document_id="auth-doc",
            chunk_index=0,
            keywords=["oauth", "authentication", "client credentials", "token"]
        ),
        DocumentChunk(
            id="chunk-2",
            content="Access Token Usage: After obtaining an access token from the /oauth/token endpoint, include it in the Authorization header of all API requests using the format: Authorization: Bearer <access_token>. The token expires after 3600 seconds.",
            document_id="auth-doc",
            chunk_index=1,
            keywords=["access token", "authorization", "bearer", "header"]
        ),
        DocumentChunk(
            id="chunk-3",
            content="Error Handling: Common error codes include 401 Unauthorized (invalid or expired token), 403 Forbidden (insufficient permissions), and 404 Not Found (invalid endpoint URL). Check the error response body for detailed error messages.",
            document_id="error-doc",
            chunk_index=0,
            keywords=["error", "status code", "401", "403", "404"]
        ),
        DocumentChunk(
            id="chunk-4",
            content="Token Refresh: If your access token expires, you can obtain a new token using the same client credentials. There is no refresh token in the client credentials flow - simply request a new access token when needed.",
            document_id="auth-doc",
            chunk_index=2,
            keywords=["token refresh", "expired", "client credentials"]
        ),
        DocumentChunk(
            id="chunk-5",
            content="API Endpoints: All API endpoints require authentication via Bearer token in the Authorization header. Base URL is https://api.example.com/v1. Common endpoints include /users, /products, and /orders.",
            document_id="api-doc",
            chunk_index=0,
            keywords=["endpoints", "api", "base url", "authentication"]
        )
    ]
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator()
    for chunk in chunks:
        chunk.embedding = embedding_gen.generate_embedding(chunk.content)
    
    return chunks


class EmbeddingGenerator:
    """Simple embedding generator (simulated)."""
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding (simulated)."""
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return [(hash_int >> i) % 100 / 100.0 for i in range(0, 40, 4)][:10]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0.0


def demonstrate_real_world_problem():
    """Demonstrate the real-world problem and solution."""
    print("\n" + "="*70)
    print("🎯 REAL-WORLD PROBLEM: Technical API Documentation Q&A")
    print("="*70)
    
    print("\n❌ PROBLEM:")
    print("   Users ask questions in natural language:")
    print("   • 'How do I log in?'")
    print("   • 'I'm getting an error, what's wrong?'")
    print("   • 'How do I use the API?'")
    print("   ")
    print("   But documentation uses technical terminology:")
    print("   • 'OAuth 2.0 authentication'")
    print("   • 'Access token'")
    print("   • 'Error codes 401, 403, 404'")
    print("   ")
    print("   Basic RAG fails: 'log in' ≠ 'authentication' ≠ 'OAuth 2.0'")
    
    print("\n✅ SOLUTION: Index-Aware Retrieval")
    print("   • HyDE: Generate hypothetical answer, match to that")
    print("   • Query Expansion: Translate user terms to technical terms")
    print("   • Hybrid Search: Combine keyword + semantic search")
    print("   • GraphRAG: Find related chunks via graph structure")
    
    # Create documentation and QA system
    chunks = create_sample_documentation()
    embedding_gen = EmbeddingGenerator()
    qa_system = APIDocumentationQA(chunks, embedding_gen)
    
    # Test queries
    print("\n🔍 TESTING QUERIES:")
    
    queries = [
        "How do I log in?",
        "I'm getting an error, what should I do?",
        "How do I use the API?"
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        result = qa_system.query(query)
        
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Techniques used: {', '.join(result['techniques'])}")
        print(f"   Chunks retrieved: {result['chunks_used']}")
        print(f"   Sources: {len(result['sources'])} chunks")
    
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS:")
    print("   • HyDE bridges vocabulary gaps")
    print("   • Query expansion translates user terms")
    print("   • Hybrid search combines keyword + semantic")
    print("   • GraphRAG finds related concepts")
    print("="*70)


def show_comparison():
    """Show comparison: basic RAG vs index-aware retrieval."""
    print("\n" + "="*70)
    print("⚖️  BASIC RAG vs INDEX-AWARE RETRIEVAL")
    print("="*70)
    
    print("\n🔤 Basic RAG:")
    print("   Query: 'How do I log in?'")
    print("   Search: Finds chunks with 'log' or 'in'")
    print("   Result: No relevant chunks found (vocabulary mismatch)")
    print("   Problem: User terms don't match documentation terms")
    
    print("\n📊 Index-Aware Retrieval:")
    print("   Query: 'How do I log in?'")
    print("   ")
    print("   Technique 1 (HyDE):")
    print("     • Generate: 'To authenticate, use OAuth 2.0...'")
    print("     • Match chunks to hypothetical answer")
    print("     • Result: Finds authentication chunks ✓")
    print("   ")
    print("   Technique 2 (Query Expansion):")
    print("     • Expand: 'log in' → 'authentication oauth access token'")
    print("     • Match expanded query")
    print("     • Result: Finds relevant chunks ✓")
    print("   ")
    print("   Technique 3 (Hybrid Search):")
    print("     • BM25 finds: chunks with 'oauth', 'token'")
    print("     • Semantic finds: chunks about authentication")
    print("     • Combined: Better results ✓")
    print("   ")
    print("   Technique 4 (GraphRAG):")
    print("     • Initial: Authentication chunk")
    print("     • Related: Token usage, error handling chunks")
    print("     • Result: Complete answer ✓")
    
    print("\n💡 Four Techniques:")
    print("   1. HyDE: Hypothetical Document Embedding")
    print("   2. Query Expansion: Term translation and context")
    print("   3. Hybrid Search: BM25 + Semantic combination")
    print("   4. GraphRAG: Graph-based related chunk retrieval")


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 INDEX-AWARE RETRIEVAL - API DOCUMENTATION Q&A")
    print("="*70)
    
    print("\n📋 This example solves a real-world problem:")
    print("   Answering user questions in natural language")
    print("   when documentation uses technical terminology")
    print("   and answers require connecting multiple concepts")
    
    # Show comparison
    show_comparison()
    
    # Demonstrate real-world problem solving
    demonstrate_real_world_problem()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for pattern explanation")
    print("   2. Tune hybrid search weight (α) for your data")
    print("   3. Build term translation dictionary for your domain")
    print("   4. Create meaningful graph relationships")
    print("   5. Combine techniques for best results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

