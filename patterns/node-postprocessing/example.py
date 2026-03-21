"""
Node Postprocessing Pattern - Real-World Problem Solver

PROBLEM: Legal Document Q&A System
    Your RAG system retrieves legal document chunks, but they have issues:
    - Ambiguous entities: "Apple" could be company or fruit
    - Conflicting content: Different interpretations of same law
    - Obsolete information: Old regulations superseded by new ones
    - Too verbose: Large chunks with only small relevant sections
    - Low relevance: Initial retrieval ranking isn't accurate enough

SOLUTION: Node Postprocessing with Multiple Techniques
    - Reranking: Use BGE-style cross-encoder models for more accurate ranking
    - Hybrid Search: Combine BM25 (keyword) and semantic (embedding) search
    - Query Expansion: Expand queries with legal terminology
    - Filtering: Remove obsolete, conflicting, or irrelevant chunks
    - Contextual Compression: Extract only relevant parts from verbose chunks
    - Disambiguation: Resolve ambiguous entities and clarify context

This example implements a working legal document Q&A system that handles
all these issues through comprehensive node postprocessing.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import math
import re
from datetime import datetime

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of legal documentation."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    document_id: str = ""
    chunk_index: int = 0
    date: Optional[datetime] = None
    is_obsolete: bool = False
    superseded_by: Optional[str] = None
    entities: List[str] = field(default_factory=list)


# ============================================================================
# TECHNIQUE 1: RERANKING (BGE-style Cross-Encoder)
# ============================================================================

class Reranker:
    """
    Reranker using cross-encoder approach (BGE-style).
    
    More accurate than embedding models because it sees query and chunk together.
    In production: from sentence_transformers import CrossEncoder
                   model = CrossEncoder('BAAI/bge-reranker-base')
    """
    
    def rerank(self, query: str, chunks: List[DocumentChunk], 
              top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Rerank chunks using cross-encoder approach.
        
        Cross-encoders are more accurate than bi-encoders (embedding models)
        because they see query and chunk together, enabling better relevance judgment.
        """
        # Simulate cross-encoder scoring
        # In production: scores = model.predict([(query, chunk.content) for chunk in chunks])
        
        scored_chunks = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            content_words = set(content_lower.split())
            
            # Cross-encoder sees query and chunk together
            # More accurate than embedding similarity alone
            overlap = len(query_words & content_words)
            total_unique = len(query_words | content_words)
            
            # Base score from word overlap
            base_score = overlap / max(total_unique, 1)
            
            # Cross-encoder is better at phrase matching
            query_phrases = self._extract_phrases(query_lower)
            for phrase in query_phrases:
                if phrase in content_lower:
                    base_score += 0.15  # Phrase match bonus
            
            # Boost for keyword matches (cross-encoder understands context)
            if chunk.keywords:
                keyword_matches = sum(1 for kw in chunk.keywords if kw.lower() in query_lower)
                base_score += keyword_matches * 0.1
            
            # Cross-encoder understands semantic relationships better
            if self._semantic_match(query_lower, content_lower):
                base_score += 0.2
            
            scored_chunks.append((chunk, min(base_score, 1.0)))
        
        # Sort by reranking score (more accurate than initial retrieval)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_k]
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract 2-3 word phrases from text."""
        words = text.split()
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        return phrases
    
    def _semantic_match(self, query: str, content: str) -> bool:
        """Check for semantic relationships (simplified)."""
        # In production, would use actual semantic understanding
        semantic_pairs = [
            ("breach", "violation"), ("damages", "compensation"),
            ("contract", "agreement"), ("liability", "responsibility")
        ]
        for term1, term2 in semantic_pairs:
            if (term1 in query and term2 in content) or (term2 in query and term1 in content):
                return True
        return False


# ============================================================================
# TECHNIQUE 2: HYBRID SEARCH (BM25 + Semantic)
# ============================================================================

class BM25Scorer:
    """BM25 keyword-based scoring."""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.avg_doc_length = self._calculate_avg_length()
        self.k1 = 1.5
        self.b = 0.75
    
    def _calculate_avg_length(self) -> float:
        if not self.chunks:
            return 1.0
        total_length = sum(len(chunk.content.split()) for chunk in self.chunks)
        return total_length / len(self.chunks)
    
    def score(self, query: str, chunk: DocumentChunk) -> float:
        """Calculate BM25 score."""
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
                # Simplified IDF
                idf = math.log((len(self.chunks) + 1) / (1 + 1))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score += idf * (numerator / denominator)
        
        return score


class HybridRetriever:
    """
    Hybrid Search: Combines BM25 (keyword) and Semantic (embedding) search.
    
    final_score = α × BM25_score + (1-α) × semantic_score
    Typically α = 0.3-0.7 depending on data characteristics.
    """
    
    def __init__(self, chunks: List[DocumentChunk], embedding_generator, alpha: float = 0.4):
        self.chunks = chunks
        self.embedding_generator = embedding_generator
        self.alpha = alpha  # Weight for BM25 (0.4 = 40% BM25, 60% semantic)
        self.bm25_scorer = BM25Scorer(chunks)
        
        # Generate embeddings for all chunks
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = embedding_generator.generate_embedding(chunk.content)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve using hybrid search."""
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        scored_chunks = []
        for chunk in self.chunks:
            # BM25 score (keyword-based)
            bm25_score = self.bm25_scorer.score(query, chunk)
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
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]


# ============================================================================
# TECHNIQUE 3: QUERY EXPANSION AND DECOMPOSITION
# ============================================================================

class QueryProcessor:
    """
    Query Expansion and Decomposition.
    
    - Expansion: Add synonyms and related terms
    - Decomposition: Break complex queries into sub-queries
    """
    
    def __init__(self):
        # Legal term expansions
        self.term_expansions = {
            "contract": ["agreement", "covenant", "compact", "pact"],
            "liability": ["responsibility", "obligation", "accountability"],
            "breach": ["violation", "infringement", "non-compliance", "default"],
            "damages": ["compensation", "reparation", "restitution", "indemnity"],
            "party": ["participant", "signatory", "entity"],
            "obligation": ["duty", "requirement", "responsibility"]
        }
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.
        
        Translates user terms to technical terms used in documents.
        """
        expanded_terms = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.term_expansions.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        # Remove duplicates while preserving order
        return " ".join(list(dict.fromkeys(expanded_terms)))
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.
        
        Handles multi-part questions by breaking them down.
        """
        sub_queries = []
        
        # Check for multiple questions (and/or)
        if " and " in query.lower():
            parts = [p.strip() for p in query.split(" and ")]
            sub_queries.extend(parts)
        elif " or " in query.lower():
            parts = [p.strip() for p in query.split(" or ")]
            sub_queries.extend(parts)
        else:
            sub_queries.append(query)
        
        return sub_queries if sub_queries else [query]


# ============================================================================
# TECHNIQUE 4: FILTERING
# ============================================================================

class ChunkFilter:
    """
    Filter obsolete, conflicting, or irrelevant chunks.
    
    Removes problematic content before final ranking.
    """
    
    def filter_obsolete(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Filter out obsolete chunks (superseded by newer versions)."""
        filtered = []
        for chunk in chunks:
            if not chunk.is_obsolete and chunk.superseded_by is None:
                filtered.append(chunk)
        return filtered
    
    def filter_by_relevance(self, chunks: List[Tuple[DocumentChunk, float]], 
                           threshold: float = 0.3) -> List[DocumentChunk]:
        """Filter chunks below relevance threshold."""
        filtered = []
        for chunk, score in chunks:
            if score >= threshold:
                filtered.append(chunk)
        return filtered
    
    def filter_conflicting(self, chunks: List[DocumentChunk], 
                          query: str) -> List[DocumentChunk]:
        """
        Filter chunks with conflicting information.
        
        In production, would use more sophisticated conflict detection.
        """
        filtered = []
        seen_topics = set()
        
        for chunk in chunks:
            # Simple heuristic: avoid duplicate topics
            # In production, would detect actual contradictions
            topic_key = " ".join(sorted(chunk.keywords[:2]))
            if topic_key not in seen_topics or len(seen_topics) < 3:
                seen_topics.add(topic_key)
                filtered.append(chunk)
        
        return filtered


# ============================================================================
# TECHNIQUE 5: CONTEXTUAL COMPRESSION
# ============================================================================

class ContextualCompressor:
    """
    Contextual Compression: Extract relevant parts from verbose chunks.
    
    Reduces noise by extracting only query-relevant sections.
    In production, would use LLM to intelligently extract relevant parts.
    """
    
    def compress(self, chunk: DocumentChunk, query: str, 
                max_length: int = 200) -> DocumentChunk:
        """
        Compress chunk to relevant parts.
        
        Extracts sentences containing query terms, removing verbose background.
        """
        query_words = set(query.lower().split())
        sentences = [s.strip() for s in chunk.content.split('.') if s.strip()]
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            # Check if sentence has overlap with query
            if query_words & sentence_words:
                relevant_sentences.append(sentence)
        
        # Combine relevant sentences
        if relevant_sentences:
            compressed_content = '. '.join(relevant_sentences[:3])
            if not compressed_content.endswith('.'):
                compressed_content += '.'
            if len(compressed_content) > max_length:
                compressed_content = compressed_content[:max_length] + "..."
        else:
            # Fallback: first part of chunk
            compressed_content = chunk.content[:max_length] + "..."
        
        # Create compressed chunk
        compressed_chunk = DocumentChunk(
            id=chunk.id + "_compressed",
            content=compressed_content,
            embedding=chunk.embedding,
            keywords=chunk.keywords,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            date=chunk.date,
            entities=chunk.entities
        )
        
        return compressed_chunk


# ============================================================================
# TECHNIQUE 6: DISAMBIGUATION
# ============================================================================

class Disambiguator:
    """
    Disambiguation: Resolve ambiguous entities and clarify context.
    
    Handles cases where same term refers to different things.
    """
    
    def __init__(self):
        # Entity disambiguation dictionary
        self.entity_contexts = {
            "apple": {
                "company": ["technology", "iphone", "ipad", "mac", "software", "corporate", "inc"],
                "fruit": ["nutrition", "vitamin", "eating", "healthy", "food", "diet"]
            },
            "bank": {
                "financial": ["money", "loan", "account", "deposit", "interest", "financial"],
                "river": ["water", "river", "shore", "flowing", "geography", "natural"]
            }
        }
    
    def disambiguate(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """
        Disambiguate entities in chunks based on query context.
        
        Resolves ambiguous terms by analyzing context clues.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        disambiguated = []
        for chunk in chunks:
            chunk_lower = chunk.content.lower()
            
            # Check for ambiguous entities
            for entity, contexts in self.entity_contexts.items():
                if entity in chunk_lower:
                    # Determine entity type based on query and chunk context
                    entity_type = self._determine_entity_type(entity, query_words, chunk_lower)
                    if entity_type:
                        # Add context marker (in production, would modify content intelligently)
                        # For demo, we'll just track it
                        chunk.entities.append(f"{entity}:{entity_type}")
                        disambiguated.append(chunk)
                        break
            else:
                # No ambiguous entity found
                disambiguated.append(chunk)
        
        return disambiguated
    
    def _determine_entity_type(self, entity: str, query_words: set, 
                              chunk_content: str) -> Optional[str]:
        """Determine entity type based on context clues."""
        if entity not in self.entity_contexts:
            return None
        
        contexts = self.entity_contexts[entity]
        chunk_lower = chunk_content.lower()
        
        # Score each context based on presence of context keywords
        scores = {}
        for context_type, keywords in contexts.items():
            score = sum(1 for kw in keywords if kw in query_words or kw in chunk_lower)
            scores[context_type] = score
        
        # Return highest scoring context
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return None


# ============================================================================
# LEGAL DOCUMENT Q&A SYSTEM
# ============================================================================

class LegalDocumentQA:
    """
    Legal Document Q&A System with Node Postprocessing.
    
    This solves the real problem: handling ambiguous entities, conflicting
    information, obsolete regulations, and verbose chunks in legal documents.
    """
    
    def __init__(self, chunks: List[DocumentChunk], embedding_generator):
        self.chunks = chunks
        self.embedding_generator = embedding_generator
        
        # Initialize all postprocessing components
        self.reranker = Reranker()
        self.hybrid_retriever = HybridRetriever(chunks, embedding_generator)
        self.query_processor = QueryProcessor()
        self.filter = ChunkFilter()
        self.compressor = ContextualCompressor()
        self.disambiguator = Disambiguator()
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query with comprehensive node postprocessing.
        
        This solves the real problem by applying all postprocessing techniques:
        1. Query expansion and decomposition
        2. Hybrid search (initial retrieval)
        3. Filtering (obsolete, conflicting, low-relevance)
        4. Reranking (BGE-style for accuracy)
        5. Disambiguation (resolve ambiguous entities)
        6. Contextual compression (extract relevant parts)
        """
        # Step 1: Query expansion and decomposition
        expanded_query = self.query_processor.expand_query(question)
        sub_queries = self.query_processor.decompose_query(expanded_query)
        
        # Step 2: Initial retrieval using hybrid search
        all_candidates = []
        for sub_query in sub_queries:
            candidates = self.hybrid_retriever.retrieve(sub_query, top_k=10)
            all_candidates.extend(candidates)
        
        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for chunk, score in all_candidates:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_candidates.append((chunk, score))
        
        # Step 3: Filtering
        chunks_only = [chunk for chunk, _ in unique_candidates]
        
        # Filter obsolete chunks
        filtered_chunks = self.filter.filter_obsolete(chunks_only)
        
        # Filter by relevance threshold
        filtered_with_scores = [(c, s) for c, s in unique_candidates if c in filtered_chunks]
        filtered_chunks = self.filter.filter_by_relevance(filtered_with_scores, threshold=0.2)
        
        # Filter conflicting chunks
        filtered_chunks = self.filter.filter_conflicting(filtered_chunks, question)
        
        # Step 4: Reranking (more accurate than initial retrieval)
        reranked = self.reranker.rerank(question, filtered_chunks, top_k=5)
        
        # Step 5: Disambiguation
        reranked_chunks = [chunk for chunk, _ in reranked]
        disambiguated = self.disambiguator.disambiguate(reranked_chunks, question)
        
        # Step 6: Contextual compression
        compressed = []
        for chunk in disambiguated:
            compressed_chunk = self.compressor.compress(chunk, question, max_length=200)
            compressed.append(compressed_chunk)
        
        # Generate answer
        answer = self._generate_answer(compressed, question)
        
        return {
            "answer": answer,
            "chunks_used": len(compressed),
            "postprocessing_steps": [
                "Query Expansion & Decomposition",
                "Hybrid Search (BM25 + Semantic)",
                "Filtering (obsolete, conflicting, low-relevance)",
                "Reranking (BGE-style cross-encoder)",
                "Disambiguation (entity resolution)",
                "Contextual Compression (extract relevant parts)"
            ],
            "sources": [
                {
                    "content": chunk.content[:150] + "...",
                    "document": chunk.document_id,
                    "date": chunk.date.strftime("%Y-%m-%d") if chunk.date else "Unknown",
                    "entities": chunk.entities
                }
                for chunk in compressed[:3]
            ]
        }
    
    def _generate_answer(self, chunks: List[DocumentChunk], question: str) -> str:
        """Generate answer from processed chunks."""
        if not chunks:
            return "No relevant information found after postprocessing."
        
        primary = chunks[0]
        answer = f"Based on legal documentation: {primary.content}"
        
        if len(chunks) > 1:
            answer += f"\n\nAdditional context: {chunks[1].content[:100]}..."
        
        return answer


# ============================================================================
# DEMONSTRATION
# ============================================================================

class EmbeddingGenerator:
    """Simple embedding generator (simulated)."""
    
    def generate_embedding(self, text: str) -> List[float]:
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return [(hash_int >> i) % 100 / 100.0 for i in range(0, 40, 4)][:10]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0.0


def create_sample_legal_docs() -> List[DocumentChunk]:
    """Create sample legal document chunks with various issues."""
    chunks = [
        DocumentChunk(
            id="chunk-1",
            content="Contract Breach and Liability: When a party fails to perform their obligations under a contract, this constitutes a breach. The non-breaching party may seek damages including compensatory damages for losses incurred, consequential damages for indirect losses, and in some cases punitive damages. The statute of limitations for breach of contract claims is typically three years from the date of breach.",
            document_id="contract-law-2023",
            chunk_index=0,
            date=datetime(2023, 1, 15),
            keywords=["contract", "breach", "liability", "damages", "statute"],
            entities=["contract", "party"]
        ),
        DocumentChunk(
            id="chunk-2",
            content="Outdated Regulation: The previous regulation from 2010 stated that contract breaches must be reported within 30 days. This regulation has been superseded by the 2023 Contract Law Act which extends the reporting period to 90 days. The old 30-day requirement is no longer valid.",
            document_id="old-regulation-2010",
            chunk_index=0,
            date=datetime(2010, 5, 10),
            is_obsolete=True,
            superseded_by="contract-law-2023",
            keywords=["regulation", "contract", "breach", "reporting", "30 days"],
            entities=["regulation"]
        ),
        DocumentChunk(
            id="chunk-3",
            content="Apple Inc. Corporate Liability: Technology companies like Apple Inc. are subject to product liability laws. If a product defect causes harm, the company may be held liable for damages. This includes both compensatory and punitive damages depending on the severity of the defect and the company's knowledge of the issue. Apple Inc. has faced numerous product liability cases related to device defects.",
            document_id="corporate-law-2023",
            chunk_index=0,
            date=datetime(2023, 3, 20),
            keywords=["apple", "liability", "product", "damages", "corporate"],
            entities=["Apple Inc.", "company"]
        ),
        DocumentChunk(
            id="chunk-4",
            content="Conflicting Interpretation: Some legal scholars argue that breach of contract should only result in compensatory damages, while others maintain that punitive damages are appropriate in cases of willful breach. The courts have generally favored compensatory damages except in cases of fraud or malicious intent. This creates uncertainty in contract law interpretation.",
            document_id="legal-scholarship-2022",
            chunk_index=0,
            date=datetime(2022, 8, 15),
            keywords=["breach", "damages", "compensatory", "punitive", "interpretation"],
            entities=["scholars", "courts"]
        ),
        DocumentChunk(
            id="chunk-5",
            content="Verbose Legal Text: This section contains extensive background information about the history of contract law, including references to Roman law, common law developments, and modern statutory frameworks. The evolution of contract law spans centuries, from ancient Roman contracts to medieval merchant agreements to modern commercial contracts. Various legal systems have influenced contract law, including civil law traditions and common law principles. The key point relevant to breach of contract is that modern law allows for both compensatory and consequential damages, with punitive damages reserved for exceptional cases involving fraud or malicious conduct.",
            document_id="legal-history-2023",
            chunk_index=0,
            date=datetime(2023, 6, 10),
            keywords=["contract", "law", "history", "damages", "evolution"],
            entities=["law"]
        ),
        DocumentChunk(
            id="chunk-6",
            content="Low Relevance Chunk: This document discusses tax law and income reporting requirements. Taxpayers must file returns annually and report all income sources. The Internal Revenue Service enforces tax compliance through audits and penalties. This content is not relevant to contract breach questions.",
            document_id="tax-law-2023",
            chunk_index=0,
            date=datetime(2023, 2, 1),
            keywords=["tax", "income", "irs", "filing"],
            entities=["taxpayer"]
        )
    ]
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator()
    for chunk in chunks:
        chunk.embedding = embedding_gen.generate_embedding(chunk.content)
    
    return chunks


def demonstrate_real_world_problem():
    """Demonstrate the real-world problem and solution."""
    print("\n" + "="*70)
    print("🎯 REAL-WORLD PROBLEM: Legal Document Q&A System")
    print("="*70)
    
    print("\n❌ PROBLEM:")
    print("   Retrieved chunks have issues:")
    print("   • Ambiguous entities: 'Apple' could be company or fruit")
    print("   • Conflicting content: Different interpretations of same law")
    print("   • Obsolete information: Old 2010 regulation superseded by 2023 law")
    print("   • Too verbose: Large chunks with only small relevant sections")
    print("   • Low relevance: Tax law chunk retrieved for contract question")
    print("   • Low accuracy: Initial retrieval ranking isn't precise enough")
    
    print("\n✅ SOLUTION: Node Postprocessing")
    print("   • Reranking: BGE-style cross-encoder for accurate ranking")
    print("   • Hybrid Search: Combine BM25 + semantic search")
    print("   • Query Expansion: Expand with legal terminology")
    print("   • Filtering: Remove obsolete, conflicting, low-relevance chunks")
    print("   • Contextual Compression: Extract relevant parts from verbose chunks")
    print("   • Disambiguation: Resolve ambiguous entities")
    
    # Create system
    chunks = create_sample_legal_docs()
    embedding_gen = EmbeddingGenerator()
    qa_system = LegalDocumentQA(chunks, embedding_gen)
    
    # Test queries
    print("\n🔍 TESTING QUERIES:")
    
    queries = [
        "What are the damages for breach of contract?",
        "What is Apple's liability?",
        "What happens when a contract is breached?"
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        result = qa_system.query(query)
        
        print(f"   Answer: {result['answer'][:250]}...")
        print(f"   Postprocessing steps applied: {len(result['postprocessing_steps'])}")
        print(f"   Chunks after postprocessing: {result['chunks_used']}")
        if result['sources']:
            print(f"   Top source: {result['sources'][0]['document']} ({result['sources'][0]['date']})")
            if result['sources'][0]['entities']:
                print(f"   Entities: {result['sources'][0]['entities']}")
    
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS:")
    print("   • Reranking improves accuracy over initial retrieval")
    print("   • Filtering removes obsolete and conflicting content")
    print("   • Compression extracts relevant parts from verbose chunks")
    print("   • Disambiguation resolves ambiguous entities")
    print("   • All techniques work together for best results")
    print("="*70)


def show_comparison():
    """Show comparison: basic retrieval vs node postprocessing."""
    print("\n" + "="*70)
    print("⚖️  BASIC RETRIEVAL vs NODE POSTPROCESSING")
    print("="*70)
    
    print("\n🔤 Basic Retrieval:")
    print("   Query: 'What are damages for breach of contract?'")
    print("   Retrieval: Finds chunks with 'damages', 'breach', 'contract'")
    print("   Problems:")
    print("     • Returns obsolete 2010 regulation (should be filtered)")
    print("     • Returns conflicting interpretations (creates confusion)")
    print("     • Returns verbose chunks with irrelevant history")
    print("     • Returns low-relevance tax law chunk")
    print("     • Ranking may not be accurate (embedding similarity only)")
    
    print("\n📊 Node Postprocessing:")
    print("   Query: 'What are damages for breach of contract?'")
    print("   ")
    print("   Step 1: Query Expansion")
    print("     • 'breach' → 'breach violation infringement non-compliance'")
    print("     • 'damages' → 'damages compensation reparation restitution'")
    print("     • Better vocabulary matching ✓")
    print("   ")
    print("   Step 2: Hybrid Search")
    print("     • BM25 finds: Keyword matches ('breach', 'contract')")
    print("     • Semantic finds: Conceptual matches (liability, obligations)")
    print("     • Combined: Better coverage ✓")
    print("   ")
    print("   Step 3: Filtering")
    print("     • Removes: Obsolete 2010 regulation ✓")
    print("     • Removes: Low-relevance tax law chunk ✓")
    print("     • Removes: Some conflicting interpretations ✓")
    print("   ")
    print("   Step 4: Reranking (BGE-style)")
    print("     • Cross-encoder sees query + chunk together")
    print("     • More accurate than embedding similarity alone")
    print("     • Better phrase matching and context understanding ✓")
    print("   ")
    print("   Step 5: Disambiguation")
    print("     • Resolves: 'Apple' → 'Apple Inc.' (company context)")
    print("     • Clarifies entity meanings ✓")
    print("   ")
    print("   Step 6: Contextual Compression")
    print("     • Extracts: Only sentences about damages and breach")
    print("     • Removes: Verbose history and background")
    print("     • Focused, relevant content ✓")
    
    print("\n💡 Six Postprocessing Techniques:")
    print("   1. Reranking: BGE-style cross-encoder models (more accurate)")
    print("   2. Hybrid Search: BM25 + Semantic combination (better coverage)")
    print("   3. Query Expansion: Term expansion and decomposition (vocabulary matching)")
    print("   4. Filtering: Obsolete, conflicting, low-relevance removal (quality)")
    print("   5. Contextual Compression: Extract relevant parts (focus)")
    print("   6. Disambiguation: Resolve ambiguous entities (clarity)")


def show_technique_details():
    """Show details of each technique."""
    print("\n" + "="*70)
    print("🔧 POSTPROCESSING TECHNIQUES IN DETAIL")
    print("="*70)
    
    print("\n1️⃣  RERANKING (BGE-style)")
    print("   • Cross-encoder models see query + chunk together")
    print("   • More accurate than bi-encoder (embedding) models")
    print("   • Better at phrase matching and context understanding")
    print("   • Models: BGE-reranker-base, BGE-reranker-large")
    print("   • Tradeoff: Slower than embeddings, but more accurate")
    
    print("\n2️⃣  HYBRID SEARCH")
    print("   • Combines BM25 (keyword) and semantic (embedding) search")
    print("   • Formula: score = α × BM25 + (1-α) × semantic")
    print("   • Typically α = 0.3-0.7 depending on data")
    print("   • BM25: Good for exact keyword matches")
    print("   • Semantic: Good for conceptual matches")
    print("   • Together: Best of both worlds")
    
    print("\n3️⃣  QUERY EXPANSION & DECOMPOSITION")
    print("   • Expansion: Add synonyms and related terms")
    print("   • Decomposition: Break complex queries into sub-queries")
    print("   • Translates user terms to document terminology")
    print("   • Handles multi-part questions")
    
    print("\n4️⃣  FILTERING")
    print("   • Obsolete: Remove superseded/outdated content")
    print("   • Conflicting: Remove contradictory information")
    print("   • Relevance: Remove chunks below threshold")
    print("   • Quality: Ensures only good content is used")
    
    print("\n5️⃣  CONTEXTUAL COMPRESSION")
    print("   • Extract only relevant sentences from verbose chunks")
    print("   • Remove background and irrelevant information")
    print("   • Reduces noise, saves tokens, improves focus")
    print("   • Can use LLM to intelligently extract relevant parts")
    
    print("\n6️⃣  DISAMBIGUATION")
    print("   • Resolve ambiguous entities (Apple = company or fruit?)")
    print("   • Use context clues from query and chunk")
    print("   • Entity linking and context analysis")
    print("   • Ensures correct interpretation")


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 NODE POSTPROCESSING - LEGAL DOCUMENT Q&A")
    print("="*70)
    
    print("\n📋 This example solves a real-world problem:")
    print("   Handling ambiguous entities, conflicting information,")
    print("   obsolete regulations, and verbose chunks in legal documents")
    print("   through comprehensive node postprocessing")
    
    # Show technique details
    show_technique_details()
    
    # Show comparison
    show_comparison()
    
    # Demonstrate real-world problem solving
    demonstrate_real_world_problem()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for detailed pattern explanation")
    print("   2. Use BGE-reranker models for production reranking")
    print("   3. Tune hybrid search weight (α) for your data")
    print("   4. Build domain-specific term expansion dictionaries")
    print("   5. Set appropriate filtering thresholds")
    print("   6. Combine all techniques for best results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
