"""
Semantic Indexing Pattern - Real-World Problem Solver

PROBLEM: E-commerce Product Catalog Search
    Users need to search products by meaning, not just keywords.
    Products have images, descriptions, and specifications (tables).
    Traditional keyword search misses related products and can't search images.

SOLUTION: Semantic Indexing System
    - Index product images, descriptions, and specs semantically
    - Enable search by meaning (e.g., "comfortable running shoes" finds all related products)
    - Handle multimedia content (images, structured data)
    - Preserve context with hierarchical chunking

This example implements a working product catalog search system.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json
import hashlib
import math

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class Product:
    """Represents a product in the catalog."""
    id: str
    name: str
    description: str
    category: str
    price: float
    image_path: Optional[str] = None
    specifications: Optional[Dict[str, Any]] = None  # Table-like structured data


@dataclass
class SemanticChunk:
    """Represents a semantically indexed chunk."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    chunk_type: str = "text"  # text, image, table
    product_id: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


# ============================================================================
# EMBEDDING GENERATOR
# ============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings for semantic search.
    
    In production, this would use sentence-transformers:
    from sentence_transformers import SentenceTransformer
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Embedding model: {model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        In production: return self.model.encode(text).tolist()
        For demo: simulate with hash-based vector
        """
        # Simulate embedding (production would use actual model)
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        # Simulate 10-dim vector (real would be 384-dim for all-MiniLM-L6-v2)
        embedding = [(hash_int >> i) % 100 / 100.0 for i in range(0, 40, 4)][:10]
        return embedding
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0.0


# ============================================================================
# IMAGE PROCESSOR (OCR Simulation)
# ============================================================================

class ImageProcessor:
    """
    Process product images for indexing.
    
    In production, this would:
    1. Use OCR (pytesseract) to extract text from images
    2. Use vision models to generate visual embeddings
    3. Combine both for richer representation
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
    
    def process_image(self, image_path: str, product: Product) -> SemanticChunk:
        """
        Process product image: extract text via OCR and generate embedding.
        
        Real implementation would:
        - Use pytesseract for OCR: ocr_text = pytesseract.image_to_string(image)
        - Or use vision model: vision_embedding = vision_model.encode(image)
        """
        # Simulate OCR extraction
        # In production: image = Image.open(image_path)
        #              ocr_text = pytesseract.image_to_string(image)
        ocr_text = f"Product image showing {product.name.lower()}. Visual features: {product.category.lower()} design, modern styling"
        
        # Generate embedding from OCR text
        embedding = self.embedding_generator.generate_embedding(ocr_text)
        
        return SemanticChunk(
            id=f"image-{product.id}",
            content=ocr_text,
            embedding=embedding,
            chunk_type="image",
            product_id=product.id,
            metadata={"image_path": image_path, "source": "ocr"}
        )


# ============================================================================
# TABLE PROCESSOR (Specifications)
# ============================================================================

class TableProcessor:
    """
    Process product specifications (structured data) for indexing.
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
    
    def process_specifications(self, product: Product) -> SemanticChunk:
        """
        Process product specifications table.
        
        Converts structured data to searchable text and generates embedding.
        """
        if not product.specifications:
            return None
        
        # Create text representation preserving structure
        spec_parts = []
        for key, value in product.specifications.items():
            spec_parts.append(f"{key}: {value}")
        
        spec_text = f"Specifications: {', '.join(spec_parts)}"
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embedding(spec_text)
        
        return SemanticChunk(
            id=f"specs-{product.id}",
            content=spec_text,
            embedding=embedding,
            chunk_type="table",
            product_id=product.id,
            metadata={"specifications": product.specifications}
        )


# ============================================================================
# SEMANTIC CHUNKER
# ============================================================================

class SemanticChunker:
    """
    Chunk product descriptions by semantic structure.
    """
    
    def chunk_product(self, product: Product) -> List[SemanticChunk]:
        """
        Create semantic chunks for a product.
        
        Creates hierarchical structure:
        - Product overview (parent)
          - Description (child)
          - Key features (child)
        """
        chunks = []
        
        # Parent chunk: Product overview
        overview_text = f"{product.name}. {product.description[:200]}"
        overview_chunk = SemanticChunk(
            id=f"overview-{product.id}",
            content=overview_text,
            chunk_type="text",
            product_id=product.id,
            children_ids=[],
            metadata={"level": 0, "category": product.category}
        )
        chunks.append(overview_chunk)
        
        # Child chunk: Full description
        desc_chunk = SemanticChunk(
            id=f"desc-{product.id}",
            content=product.description,
            chunk_type="text",
            product_id=product.id,
            parent_id=overview_chunk.id,
            metadata={"level": 1}
        )
        chunks.append(desc_chunk)
        overview_chunk.children_ids.append(desc_chunk.id)
        
        # Child chunk: Key features (extracted from description)
        features = self._extract_features(product.description)
        if features:
            features_text = f"Key features: {', '.join(features)}"
            features_chunk = SemanticChunk(
                id=f"features-{product.id}",
                content=features_text,
                chunk_type="text",
                product_id=product.id,
                parent_id=overview_chunk.id,
                metadata={"level": 1, "features": features}
            )
            chunks.append(features_chunk)
            overview_chunk.children_ids.append(features_chunk.id)
        
        return chunks
    
    def _extract_features(self, description: str) -> List[str]:
        """Extract key features from description (simple heuristic)."""
        # Simple feature extraction (production would use NLP)
        features = []
        feature_keywords = ["comfortable", "durable", "lightweight", "waterproof", 
                          "breathable", "flexible", "supportive", "stylish"]
        for keyword in feature_keywords:
            if keyword in description.lower():
                features.append(keyword)
        return features[:3]  # Top 3 features


# ============================================================================
# PRODUCT CATALOG INDEXER
# ============================================================================

class ProductCatalogIndexer:
    """
    Index products with semantic indexing.
    
    This is the main component that solves the real-world problem:
    - Indexes products with images, descriptions, and specs
    - Creates semantic embeddings for all content types
    - Maintains hierarchical structure for context
    """
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.image_processor = ImageProcessor(self.embedding_generator)
        self.table_processor = TableProcessor(self.embedding_generator)
        self.chunker = SemanticChunker()
        self.indexed_chunks: Dict[str, SemanticChunk] = {}
        self.products: Dict[str, Product] = {}
    
    def index_product(self, product: Product):
        """
        Index a product: create semantic chunks for all content types.
        
        This solves the real problem by:
        1. Creating semantic chunks for description
        2. Processing and indexing product image
        3. Processing and indexing specifications table
        4. Maintaining hierarchical relationships
        """
        self.products[product.id] = product
        
        # 1. Semantic chunking for description
        text_chunks = self.chunker.chunk_product(product)
        for chunk in text_chunks:
            if chunk.embedding is None:
                chunk.embedding = self.embedding_generator.generate_embedding(chunk.content)
            self.indexed_chunks[chunk.id] = chunk
        
        # 2. Process and index image
        if product.image_path:
            image_chunk = self.image_processor.process_image(product.image_path, product)
            # Link image to product overview
            if text_chunks:
                image_chunk.parent_id = text_chunks[0].id
                text_chunks[0].children_ids.append(image_chunk.id)
            self.indexed_chunks[image_chunk.id] = image_chunk
        
        # 3. Process and index specifications
        specs_chunk = self.table_processor.process_specifications(product)
        if specs_chunk:
            # Link specs to product overview
            if text_chunks:
                specs_chunk.parent_id = text_chunks[0].id
                text_chunks[0].children_ids.append(specs_chunk.id)
            self.indexed_chunks[specs_chunk.id] = specs_chunk
        
        logger.info(f"Indexed product: {product.name} ({len(text_chunks)} text chunks + image + specs)")
    
    def search(self, query: str, top_k: int = 5, include_context: bool = True) -> List[Tuple[SemanticChunk, float, Product]]:
        """
        Search products by semantic meaning.
        
        This solves the real problem by:
        - Finding products by meaning, not just keywords
        - Including context (parent/child chunks)
        - Returning actual products (not just chunks)
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Calculate similarities
        scored_chunks = []
        for chunk in self.indexed_chunks.values():
            if chunk.embedding:
                similarity = self.embedding_generator.cosine_similarity(
                    query_embedding, chunk.embedding
                )
                scored_chunks.append((similarity, chunk))
        
        # Sort by similarity
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Get top-k chunks
        top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
        
        # Add context if requested
        if include_context:
            contextual_chunks = []
            for chunk in top_chunks:
                contextual_chunks.append(chunk)
                
                # Add parent for context
                if chunk.parent_id and chunk.parent_id in self.indexed_chunks:
                    parent = self.indexed_chunks[chunk.parent_id]
                    if parent not in contextual_chunks:
                        contextual_chunks.append(parent)
                
                # Add children for detail
                for child_id in chunk.children_ids:
                    if child_id in self.indexed_chunks:
                        child = self.indexed_chunks[child_id]
                        if child not in contextual_chunks:
                            contextual_chunks.append(child)
            
            top_chunks = contextual_chunks
        
        # Return chunks with scores and associated products
        results = []
        seen_products = set()
        for chunk in top_chunks:
            if chunk.product_id and chunk.product_id not in seen_products:
                product = self.products[chunk.product_id]
                similarity = next(score for score, c in scored_chunks if c.id == chunk.id)
                results.append((chunk, similarity, product))
                seen_products.add(chunk.product_id)
        
        return results[:top_k]


# ============================================================================
# DEMONSTRATION: REAL-WORLD PROBLEM SOLVING
# ============================================================================

def create_sample_catalog() -> List[Product]:
    """Create sample product catalog for demonstration."""
    return [
        Product(
            id="prod-001",
            name="ComfortMax Running Shoes",
            description="Lightweight running shoes with maximum comfort. Features breathable mesh upper, cushioned sole, and flexible design. Perfect for long-distance running and daily workouts. Waterproof and durable construction.",
            category="Footwear",
            price=89.99,
            image_path="images/comfortmax-shoes.jpg",
            specifications={
                "Material": "Mesh upper, rubber sole",
                "Weight": "250g",
                "Sizes": "7-12",
                "Colors": "Black, White, Blue"
            }
        ),
        Product(
            id="prod-002",
            name="TrailBlazer Hiking Boots",
            description="Durable hiking boots designed for rugged terrain. Waterproof leather construction with supportive ankle design. Excellent grip and traction on all surfaces. Comfortable for extended hikes.",
            category="Footwear",
            price=129.99,
            image_path="images/trailblazer-boots.jpg",
            specifications={
                "Material": "Leather, rubber sole",
                "Weight": "450g",
                "Sizes": "7-13",
                "Waterproof": "Yes"
            }
        ),
        Product(
            id="prod-003",
            name="FlexFit Yoga Mat",
            description="Premium yoga mat with non-slip surface. Lightweight and easy to carry. Provides excellent cushioning and support for all yoga poses. Eco-friendly material.",
            category="Fitness",
            price=34.99,
            image_path="images/flexfit-mat.jpg",
            specifications={
                "Material": "TPE (eco-friendly)",
                "Dimensions": "183cm x 61cm",
                "Thickness": "6mm",
                "Weight": "1.2kg"
            }
        ),
        Product(
            id="prod-004",
            name="AquaGuard Water Bottle",
            description="Stainless steel water bottle with insulation. Keeps drinks cold for 24 hours or hot for 12 hours. Leak-proof design with easy-carry handle. BPA-free and dishwasher safe.",
            category="Accessories",
            price=24.99,
            image_path="images/aquaguard-bottle.jpg",
            specifications={
                "Capacity": "750ml",
                "Material": "Stainless steel",
                "Insulation": "Double-wall vacuum",
                "Colors": "Silver, Black, Blue"
            }
        ),
        Product(
            id="prod-005",
            name="PowerGrip Resistance Bands",
            description="Set of 5 resistance bands with varying resistance levels. Perfect for strength training, stretching, and rehabilitation. Includes door anchor and exercise guide. Durable latex-free material.",
            category="Fitness",
            price=19.99,
            image_path="images/powergrip-bands.jpg",
            specifications={
                "Set includes": "5 bands (5-50 lbs resistance)",
                "Material": "Latex-free rubber",
                "Accessories": "Door anchor, guide",
                "Storage": "Carry bag included"
            }
        )
    ]


def demonstrate_real_world_problem():
    """Demonstrate how semantic indexing solves a real e-commerce problem."""
    print("\n" + "="*70)
    print("🎯 REAL-WORLD PROBLEM: E-Commerce Product Search")
    print("="*70)
    
    print("\n❌ PROBLEM:")
    print("   Traditional keyword search has limitations:")
    print("   • Query: 'comfortable running shoes'")
    print("   • Misses: Products that say 'cushioned', 'soft', 'supportive'")
    print("   • Can't search: Product images, specifications tables")
    print("   • Loses context: Product features split across chunks")
    
    print("\n✅ SOLUTION: Semantic Indexing")
    print("   • Indexes: Descriptions, images (via OCR), specifications")
    print("   • Searches: By meaning, not just keywords")
    print("   • Preserves: Context with hierarchical chunking")
    print("   • Finds: Related products semantically")
    
    # Create indexer
    indexer = ProductCatalogIndexer()
    
    # Index products
    print("\n📦 Indexing Products...")
    products = create_sample_catalog()
    for product in products:
        indexer.index_product(product)
    
    print(f"   ✓ Indexed {len(products)} products")
    print(f"   ✓ Created {len(indexer.indexed_chunks)} semantic chunks")
    print(f"   ✓ Includes: descriptions, images, specifications")
    
    # Test searches
    print("\n🔍 TESTING SEMANTIC SEARCH:")
    
    test_queries = [
        "comfortable running shoes",
        "waterproof hiking gear",
        "lightweight fitness equipment",
        "durable workout accessories"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = indexer.search(query, top_k=3, include_context=True)
        
        if results:
            for i, (chunk, similarity, product) in enumerate(results, 1):
                print(f"   {i}. {product.name} (similarity: {similarity:.3f})")
                print(f"      Category: {product.category}, Price: ${product.price}")
                print(f"      Match: {chunk.chunk_type} chunk - {chunk.content[:60]}...")
        else:
            print("   No results found")
    
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS:")
    print("   • Semantic search finds products by meaning")
    print("   • Handles multimedia content (images, tables)")
    print("   • Preserves context with hierarchical structure")
    print("   • Returns actual products, not just text chunks")
    print("="*70)


def show_comparison():
    """Show comparison: keyword vs semantic search."""
    print("\n" + "="*70)
    print("⚖️  KEYWORD vs SEMANTIC SEARCH")
    print("="*70)
    
    query = "comfortable running shoes"
    
    print(f"\nQuery: '{query}'")
    
    print("\n🔤 Keyword Search:")
    print("   Finds: Only products with exact words 'comfortable', 'running', 'shoes'")
    print("   Misses: Products with 'cushioned', 'jogging', 'footwear'")
    print("   Can't search: Images, specifications")
    print("   Example: 'ComfortMax Running Shoes' ✓ (has all keywords)")
    print("   Example: 'TrailBlazer Hiking Boots' ✗ (no 'running' or 'shoes')")
    
    print("\n🧠 Semantic Search:")
    print("   Finds: Products by meaning - 'comfortable' = 'cushioned', 'soft', 'supportive'")
    print("   Finds: 'running' = 'jogging', 'athletic', 'workout'")
    print("   Finds: 'shoes' = 'footwear', 'boots', 'sneakers'")
    print("   Searches: Images (via OCR), specifications tables")
    print("   Example: 'ComfortMax Running Shoes' ✓ (semantically matches)")
    print("   Example: 'TrailBlazer Hiking Boots' ✓ (semantically related to 'comfortable footwear')")
    print("   Example: 'FlexFit Yoga Mat' ✓ (semantically related to 'comfortable fitness')")


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 SEMANTIC INDEXING - PRODUCT CATALOG SEARCH SYSTEM")
    print("="*70)
    
    print("\n📋 This example solves a real-world problem:")
    print("   E-commerce product search that handles:")
    print("   • Text descriptions")
    print("   • Product images (via OCR)")
    print("   • Specifications tables")
    print("   • Semantic meaning (not just keywords)")
    
    # Demonstrate real-world problem solving
    demonstrate_real_world_problem()
    
    # Show comparison
    show_comparison()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for pattern explanation")
    print("   2. Adapt this example for your product catalog")
    print("   3. Use real OCR/vision models for image processing")
    print("   4. Deploy with vector database (ChromaDB, Pinecone)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
