"""
Trustworthy Generation Pattern - Real-World Problem Solver

PROBLEM: Medical Q&A System
    Your RAG system answers medical questions, but users lose trust because:
    - System answers questions outside its knowledge domain (hallucination)
    - Answers lack citations (unverifiable)
    - System provides confident answers when retrieval failed (unreliable)
    - No warnings when information is uncertain or unsupported

SOLUTION: Trustworthy Generation with Self-RAG
    - Out-of-domain detection: Refuse to answer when KB lacks information
    - Self-RAG workflow: 6-step process to ensure trustworthiness
    - Citations: Add source citations to all factual claims
    - Warnings: Flag uncertain or unsupported information
    - Guardrails: Prevent unreliable or unsafe responses

This example implements a working medical Q&A system that builds trust
through transparency, verification, and self-reflection.
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
    """Represents a chunk of medical documentation."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    source: str = ""
    document_id: str = ""
    chunk_index: int = 0


@dataclass
class ResponseChunk:
    """Represents a section of the generated response."""
    id: str
    text: str
    needs_citation: bool = False
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    warning: Optional[str] = None


# ============================================================================
# OUT-OF-DOMAIN DETECTION
# ============================================================================

class OutOfDomainDetector:
    """
    Detects when queries are outside the knowledge base domain.
    
    Uses multiple methods:
    - Embedding distance: Query too far from any chunk
    - Zero-shot classification: Classify query as in-domain/out-of-domain
    - Domain keywords: Require domain-specific terminology
    """
    
    def __init__(self, embedding_generator, threshold: float = 0.3):
        self.embedding_generator = embedding_generator
        self.threshold = threshold  # Maximum distance for in-domain
        self.domain_keywords = [
            "medical", "health", "symptom", "treatment", "diagnosis",
            "disease", "condition", "medication", "therapy", "patient"
        ]
    
    def is_out_of_domain(self, query: str, chunks: List[DocumentChunk]) -> Tuple[bool, str]:
        """
        Check if query is out of domain.
        
        Returns:
            (is_out_of_domain, reason)
        """
        # Method 1: Embedding distance
        if chunks:
            query_embedding = self.embedding_generator.generate_embedding(query)
            min_distance = min([
                1 - self.embedding_generator.cosine_similarity(
                    query_embedding, chunk.embedding
                )
                for chunk in chunks if chunk.embedding
            ])
            
            if min_distance > self.threshold:
                return True, f"Query too far from knowledge base (distance: {min_distance:.3f})"
        
        # Method 2: Zero-shot classification (simulated)
        if not self._has_domain_keywords(query):
            return True, "Query lacks domain-specific medical terminology"
        
        # Method 3: Check if any chunks retrieved
        if not chunks:
            return True, "No relevant chunks found in knowledge base"
        
        return False, ""
    
    def _has_domain_keywords(self, query: str) -> bool:
        """Check if query contains domain-specific keywords."""
        query_lower = query.lower()
        # Medical-related terms that indicate medical domain
        medical_indicators = [
            "medical", "health", "symptom", "treatment", "diagnosis",
            "disease", "condition", "medication", "therapy", "patient",
            "blood", "pressure", "diabetes", "cold", "infection",
            "doctor", "hospital", "medicine", "prescription", "cure"
        ]
        return any(indicator in query_lower for indicator in medical_indicators)


# ============================================================================
# SELF-RAG WORKFLOW
# ============================================================================

class SelfRAGProcessor:
    """
    Self-RAG: Self-reflective RAG that checks its own work.
    
    Implements 6-step workflow:
    1. Generate initial response
    2. Chunk the response into smaller sections
    3. Check whether chunk needs citation
    4. Lookup sources
    5. Incorporate citations into response
    6. Add necessary warnings or corrections
    """
    
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
    
    def process(self, query: str, retrieved_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Process query through Self-RAG workflow.
        
        This solves the real problem: ensuring all claims are supported
        and adding citations and warnings where needed.
        """
        # Step 1: Generate initial response
        initial_response = self._generate_initial_response(query, retrieved_chunks)
        
        # Step 2: Chunk the response into smaller sections
        response_chunks = self._chunk_response(initial_response)
        
        # Step 3: Check whether chunk needs citation
        for chunk in response_chunks:
            chunk.needs_citation = self._needs_citation(chunk.text)
        
        # Step 4: Lookup sources for chunks that need citations
        for chunk in response_chunks:
            if chunk.needs_citation:
                chunk.sources = self._lookup_sources(chunk.text, retrieved_chunks)
                # Calculate confidence based on source support
                chunk.confidence = self._calculate_confidence(chunk, retrieved_chunks)
        
        # Step 5: Incorporate citations into response
        final_response = self._incorporate_citations(response_chunks)
        
        # Step 6: Add necessary warnings or corrections
        warnings = self._generate_warnings(response_chunks)
        if warnings:
            final_response += "\n\n⚠️ Warnings:\n" + "\n".join(f"  • {w}" for w in warnings)
        
        return {
            "response": final_response,
            "chunks": response_chunks,
            "warnings": warnings,
            "has_citations": any(c.sources for c in response_chunks)
        }
    
    def _generate_initial_response(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Step 1: Generate initial response from retrieved chunks.
        
        In production, would use LLM with retrieved context.
        """
        if not chunks:
            return "I don't have enough information to answer this question."
        
        # Simulate LLM generation - find most relevant chunk
        # In production: response = llm.generate(context=chunks, query=query)
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Find most relevant chunk based on keyword overlap
        best_chunk = None
        best_score = 0
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
        
        if best_chunk:
            # Use the most relevant chunk as the basis for response
            return best_chunk.content
        else:
            # Fallback: use first chunk
            return chunks[0].content
    
    def _chunk_response(self, response: str) -> List[ResponseChunk]:
        """
        Step 2: Chunk the response into smaller sections.
        
        Divides response into sentences or logical sections for verification.
        """
        # Split by sentences
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        chunks = []
        for i, sentence in enumerate(sentences):
            chunks.append(ResponseChunk(
                id=f"chunk-{i}",
                text=sentence + "."
            ))
        
        return chunks
    
    def _needs_citation(self, text: str) -> bool:
        """
        Step 3: Check whether chunk needs citation.
        
        Uses classification to determine if claim is factual and needs citation.
        In production, would use a classifier model.
        """
        # Factual claims that need citations
        factual_indicators = [
            "is", "are", "causes", "treats", "diagnoses", "recommends",
            "studies show", "research indicates", "according to"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in factual_indicators)
    
    def _lookup_sources(self, claim: str, retrieved_chunks: List[DocumentChunk]) -> List[str]:
        """
        Step 4: Lookup sources for the claim.
        
        Finds which retrieved chunks support this claim.
        """
        claim_embedding = self.embedding_generator.generate_embedding(claim)
        
        supporting_sources = []
        for chunk in retrieved_chunks:
            if chunk.embedding:
                similarity = self.embedding_generator.cosine_similarity(
                    claim_embedding, chunk.embedding
                )
                if similarity > 0.5:  # Threshold for support
                    supporting_sources.append(chunk.source or chunk.document_id)
        
        return supporting_sources
    
    def _calculate_confidence(self, chunk: ResponseChunk, 
                              retrieved_chunks: List[DocumentChunk]) -> float:
        """Calculate confidence based on source support."""
        if not chunk.sources:
            return 0.3  # Low confidence if no sources
        
        # More sources = higher confidence
        confidence = min(0.5 + (len(chunk.sources) * 0.15), 1.0)
        return confidence
    
    def _incorporate_citations(self, chunks: List[ResponseChunk]) -> str:
        """
        Step 5: Incorporate citations into response.
        
        Adds inline citations to claims that need them.
        """
        cited_sections = []
        source_map = {}  # Map source names to citation numbers
        
        for chunk in chunks:
            if chunk.needs_citation and chunk.sources:
                # Create citations
                citations = []
                for source in chunk.sources:
                    if source not in source_map:
                        source_map[source] = len(source_map) + 1
                    citations.append(f"[{source_map[source]}]")
                
                # Add citations to text
                cited_text = f"{chunk.text} {''.join(citations)}"
                cited_sections.append(cited_text)
            else:
                cited_sections.append(chunk.text)
        
        # Build final response with citations
        response = " ".join(cited_sections)
        
        # Add references section
        if source_map:
            response += "\n\nReferences:\n"
            for source, num in sorted(source_map.items(), key=lambda x: x[1]):
                response += f"  [{num}] {source}\n"
        
        return response
    
    def _generate_warnings(self, chunks: List[ResponseChunk]) -> List[str]:
        """
        Step 6: Add necessary warnings or corrections.
        
        Generates warnings for uncertain or unsupported claims.
        """
        warnings = []
        
        for chunk in chunks:
            # Low confidence warning
            if chunk.confidence < 0.5:
                warnings.append(f"Low confidence in: '{chunk.text[:50]}...'")
            
            # No sources warning
            if chunk.needs_citation and not chunk.sources:
                warnings.append(f"Unsupported claim: '{chunk.text[:50]}...' (no sources found)")
        
        return warnings


# ============================================================================
# GUARDRAILS
# ============================================================================

class GuardrailSystem:
    """
    Guardrails: Prevent generation of unsafe or unreliable content.
    
    Enforces:
    - Relevance: Answer must be relevant to query
    - Source support: Answer must be supported by sources
    - Confidence: Refuse if confidence too low
    - Safety: Prevent harmful content
    """
    
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
    
    def check(self, query: str, response: Dict[str, Any], 
             retrieved_chunks: List[DocumentChunk]) -> Tuple[bool, Optional[str]]:
        """
        Check if response passes guardrails.
        
        Returns:
            (passed, reason_if_failed)
        """
        # Guardrail 1: Must have retrieved chunks
        if not retrieved_chunks:
            return False, "No relevant information found in knowledge base"
        
        # Guardrail 2: Must have citations if response has factual claims
        if response.get("has_citations") is False:
            # Check if response has factual claims
            if self._has_factual_claims(response["response"]):
                return False, "Response contains factual claims without citations"
        
        # Guardrail 3: Confidence check
        chunks = response.get("chunks", [])
        if chunks:
            avg_confidence = sum(c.confidence for c in chunks) / len(chunks)
            if avg_confidence < self.min_confidence:
                return False, f"Average confidence too low: {avg_confidence:.2f}"
        
        # Guardrail 4: Must not have critical warnings
        warnings = response.get("warnings", [])
        critical_warnings = [w for w in warnings if "Unsupported" in w]
        if critical_warnings:
            return False, "Response contains unsupported claims"
        
        return True, None
    
    def _has_factual_claims(self, text: str) -> bool:
        """Check if text contains factual claims."""
        factual_indicators = ["is", "are", "causes", "treats", "recommends"]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in factual_indicators)


# ============================================================================
# MEDICAL Q&A SYSTEM
# ============================================================================

class MedicalQA:
    """
    Medical Q&A System with Trustworthy Generation.
    
    This solves the real problem: building user trust through:
    - Out-of-domain detection
    - Self-RAG workflow
    - Citations
    - Warnings
    - Guardrails
    """
    
    def __init__(self, chunks: List[DocumentChunk], embedding_generator):
        self.chunks = chunks
        self.embedding_generator = embedding_generator
        
        # Initialize components
        self.out_of_domain_detector = OutOfDomainDetector(embedding_generator)
        self.self_rag = SelfRAGProcessor(embedding_generator)
        self.guardrails = GuardrailSystem()
        
        # Generate embeddings for all chunks
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = embedding_generator.generate_embedding(chunk.content)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query with trustworthy generation.
        
        This solves the real problem by:
        1. Detecting out-of-domain queries
        2. Applying Self-RAG workflow
        3. Adding citations and warnings
        4. Enforcing guardrails
        """
        # Retrieve relevant chunks
        retrieved = self._retrieve(question, top_k=5)
        
        # Check out-of-domain
        is_ood, ood_reason = self.out_of_domain_detector.is_out_of_domain(question, retrieved)
        if is_ood:
            return {
                "response": f"I cannot answer this question. {ood_reason}",
                "out_of_domain": True,
                "reason": ood_reason,
                "citations": [],
                "warnings": ["Query is outside knowledge domain"]
            }
        
        # Apply Self-RAG workflow
        result = self.self_rag.process(question, retrieved)
        
        # Check guardrails
        passed, guardrail_reason = self.guardrails.check(question, result, retrieved)
        if not passed:
            result["response"] = f"I cannot provide a reliable answer. {guardrail_reason}"
            result["warnings"].append(f"Guardrail: {guardrail_reason}")
        
        result["out_of_domain"] = False
        result["guardrails_passed"] = passed
        
        return result
    
    def _retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve relevant chunks."""
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        scored_chunks = []
        for chunk in self.chunks:
            if chunk.embedding:
                similarity = self.embedding_generator.cosine_similarity(
                    query_embedding, chunk.embedding
                )
                scored_chunks.append((similarity, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]


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


def create_sample_medical_docs() -> List[DocumentChunk]:
    """Create sample medical documentation chunks."""
    chunks = [
        DocumentChunk(
            id="chunk-1",
            content="Hypertension Treatment: High blood pressure is typically treated with lifestyle modifications including diet changes, exercise, and stress reduction. Medications such as ACE inhibitors, beta-blockers, and diuretics may be prescribed when lifestyle changes are insufficient. Regular monitoring is essential.",
            source="Cardiology Guidelines 2023",
            document_id="cardio-doc",
            chunk_index=0
        ),
        DocumentChunk(
            id="chunk-2",
            content="Diabetes Management: Type 2 diabetes management focuses on blood glucose control through diet, exercise, and medication. Metformin is often the first-line medication. Regular blood sugar monitoring and A1C testing are recommended every 3-6 months.",
            source="Endocrinology Guidelines 2023",
            document_id="endo-doc",
            chunk_index=0
        ),
        DocumentChunk(
            id="chunk-3",
            content="Common Cold Symptoms: The common cold typically presents with runny nose, sneezing, sore throat, and mild cough. Symptoms usually resolve within 7-10 days. Rest, hydration, and over-the-counter symptom relief medications can help manage symptoms.",
            source="General Medicine Guide 2023",
            document_id="general-doc",
            chunk_index=0
        ),
        DocumentChunk(
            id="chunk-4",
            content="Antibiotic Use: Antibiotics should only be used for bacterial infections, not viral infections like the common cold. Overuse of antibiotics contributes to antibiotic resistance. Always complete the full course of antibiotics as prescribed.",
            source="Infectious Disease Guidelines 2023",
            document_id="infectious-doc",
            chunk_index=0
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
    print("🎯 REAL-WORLD PROBLEM: Medical Q&A System")
    print("="*70)
    
    print("\n❌ PROBLEM:")
    print("   Users lose trust because:")
    print("   • System answers questions outside knowledge domain (hallucination)")
    print("   • Answers lack citations (unverifiable)")
    print("   • System provides confident answers when retrieval failed (unreliable)")
    print("   • No warnings when information is uncertain or unsupported")
    print("   • Can't verify where information came from")
    
    print("\n✅ SOLUTION: Trustworthy Generation")
    print("   • Out-of-domain detection: Refuse when KB lacks information")
    print("   • Self-RAG workflow: 6-step process to ensure trustworthiness")
    print("   • Citations: Add source citations to all factual claims")
    print("   • Warnings: Flag uncertain or unsupported information")
    print("   • Guardrails: Prevent unreliable or unsafe responses")
    
    # Create system
    chunks = create_sample_medical_docs()
    embedding_gen = EmbeddingGenerator()
    qa_system = MedicalQA(chunks, embedding_gen)
    
    # Test queries
    print("\n🔍 TESTING QUERIES:")
    
    queries = [
        "How is high blood pressure treated?",
        "What are the symptoms of common cold?",
        "How do I cook pasta?",  # Out-of-domain
        "What causes diabetes?"
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        result = qa_system.query(query)
        
        if result.get("out_of_domain"):
            print(f"   ❌ Out of Domain: {result['reason']}")
        else:
            print(f"   ✅ Response: {result['response'][:200]}...")
            print(f"   Citations: {result.get('has_citations', False)}")
            if result.get("warnings"):
                print(f"   Warnings: {len(result['warnings'])}")
                for warning in result['warnings'][:2]:
                    print(f"      • {warning}")
    
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS:")
    print("   • Out-of-domain detection prevents hallucination")
    print("   • Self-RAG ensures all claims are verified")
    print("   • Citations enable verification")
    print("   • Warnings flag uncertainty")
    print("   • Guardrails prevent unreliable responses")
    print("="*70)


def show_self_rag_workflow():
    """Show the 6-step Self-RAG workflow."""
    print("\n" + "="*70)
    print("🔄 SELF-RAG WORKFLOW (6 Steps)")
    print("="*70)
    
    print("\nSTEP 1: Generate Initial Response")
    print("   • Create draft answer from retrieved chunks")
    print("   • Use LLM to generate response based on context")
    print("   • Example: 'High blood pressure is treated with medications...'")
    
    print("\nSTEP 2: Chunk the Response")
    print("   • Divide response into smaller sections (sentences)")
    print("   • Each chunk can be independently verified")
    print("   • Example:")
    print("     Chunk 1: 'High blood pressure is treated with medications.'")
    print("     Chunk 2: 'ACE inhibitors are commonly prescribed.'")
    
    print("\nSTEP 3: Check Whether Chunk Needs Citation")
    print("   • Use classification to identify factual claims")
    print("   • Factual claims need citations")
    print("   • Example: 'is treated' → needs citation")
    
    print("\nSTEP 4: Lookup Sources")
    print("   • Find which retrieved chunks support each claim")
    print("   • Use semantic similarity to match claims to sources")
    print("   • Example: Claim about 'medications' → matches 'Cardiology Guidelines'")
    
    print("\nSTEP 5: Incorporate Citations into Response")
    print("   • Add inline citations to claims")
    print("   • Format: 'High blood pressure is treated with medications [1].'")
    print("   • Add references section at end")
    
    print("\nSTEP 6: Add Necessary Warnings or Corrections")
    print("   • Flag unsupported claims")
    print("   • Warn about low confidence")
    print("   • Example: '⚠️ Warning: Low confidence in medication dosages'")
    
    print("\n💡 Result: Trustworthy response with citations and warnings")


def show_comparison():
    """Show comparison: basic RAG vs trustworthy generation."""
    print("\n" + "="*70)
    print("⚖️  BASIC RAG vs TRUSTWORTHY GENERATION")
    print("="*70)
    
    print("\n🔤 Basic RAG:")
    print("   Query: 'How do I cook pasta?'")
    print("   Retrieval: No relevant chunks (medical KB)")
    print("   Response: 'Pasta is cooked by boiling water...' (HALLUCINATION)")
    print("   Problems:")
    print("     • Answers out-of-domain questions")
    print("     • No citations")
    print("     • No warnings")
    print("     • Can't verify information")
    
    print("\n📊 Trustworthy Generation:")
    print("   Query: 'How do I cook pasta?'")
    print("   ")
    print("   Step 1: Out-of-Domain Detection")
    print("     • Detects: Query lacks medical terminology")
    print("     • Detects: No relevant chunks found")
    print("     • Result: Refuses to answer ✓")
    print("   ")
    print("   Query: 'How is high blood pressure treated?'")
    print("   ")
    print("   Step 1: Generate Initial Response")
    print("     • 'High blood pressure is treated with medications...'")
    print("   ")
    print("   Step 2: Chunk Response")
    print("     • Chunk 1: 'High blood pressure is treated with medications.'")
    print("     • Chunk 2: 'ACE inhibitors are commonly prescribed.'")
    print("   ")
    print("   Step 3: Check Citation Needs")
    print("     • Chunk 1: Needs citation (factual claim)")
    print("     • Chunk 2: Needs citation (factual claim)")
    print("   ")
    print("   Step 4: Lookup Sources")
    print("     • Chunk 1 → Cardiology Guidelines 2023")
    print("     • Chunk 2 → Cardiology Guidelines 2023")
    print("   ")
    print("   Step 5: Incorporate Citations")
    print("     • 'High blood pressure is treated with medications [1].'")
    print("     • 'ACE inhibitors are commonly prescribed [1].'")
    print("     • References: [1] Cardiology Guidelines 2023")
    print("   ")
    print("   Step 6: Add Warnings")
    print("     • (None if all claims are supported)")
    print("   ")
    print("   Result: Trustworthy response with citations ✓")


def show_techniques():
    """Show all trustworthy generation techniques."""
    print("\n" + "="*70)
    print("🛡️  TRUSTWORTHY GENERATION TECHNIQUES")
    print("="*70)
    
    print("\n1️⃣  Out-of-Domain Detection")
    print("   • Embedding distance: Measure query-chunk similarity")
    print("   • Zero-shot classification: Categorize queries")
    print("   • Domain keywords: Require domain terminology")
    print("   • Result: Refuse to answer when KB lacks information")
    
    print("\n2️⃣  Citations")
    print("   • Source-level: Cite entire sources")
    print("   • Classification-based: Determine what needs citations")
    print("   • Token-level: Attribute each claim to source")
    print("   • Result: All factual claims are verifiable")
    
    print("\n3️⃣  Self-RAG Workflow")
    print("   • 6-step process: Generate → Chunk → Check → Lookup → Cite → Warn")
    print("   • Self-verification: System checks its own work")
    print("   • Result: Catches errors and ensures support")
    
    print("\n4️⃣  Guardrails")
    print("   • Relevance: Answer must be relevant")
    print("   • Source support: Must be supported by sources")
    print("   • Confidence: Refuse if confidence too low")
    print("   • Result: Prevents unreliable responses")
    
    print("\n5️⃣  Observability")
    print("   • Monitor citation rates")
    print("   • Track out-of-domain rates")
    print("   • Collect user feedback")
    print("   • Result: Continuous improvement")
    
    print("\n6️⃣  Human Feedback")
    print("   • Incorporate user corrections")
    print("   • Learn from mistakes")
    print("   • Improve over time")
    print("   • Result: System gets better with use")


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 TRUSTWORTHY GENERATION - MEDICAL Q&A SYSTEM")
    print("="*70)
    
    print("\n📋 This example solves a real-world problem:")
    print("   Building user trust in medical Q&A through:")
    print("   • Out-of-domain detection")
    print("   • Self-RAG workflow (6 steps)")
    print("   • Citations and source tracking")
    print("   • Warnings and guardrails")
    
    # Show techniques
    show_techniques()
    
    # Show Self-RAG workflow
    show_self_rag_workflow()
    
    # Show comparison
    show_comparison()
    
    # Demonstrate real-world problem solving
    demonstrate_real_world_problem()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for detailed pattern explanation")
    print("   2. Implement out-of-domain detection with proper thresholds")
    print("   3. Use Self-RAG workflow for all responses")
    print("   4. Add citations to all factual claims")
    print("   5. Set up guardrails and observability")
    print("   6. Incorporate human feedback for continuous improvement")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

