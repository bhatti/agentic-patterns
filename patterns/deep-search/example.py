#!/usr/bin/env python3
"""
Deep Search Pattern - Market Research Analyst System

This example implements a Deep Search system that conducts comprehensive
market research through iterative retrieval and reasoning. It demonstrates
how to handle complex information needs that basic RAG cannot satisfy.

Real-World Problem:
-------------------
Investment analysts need comprehensive research on companies/industries.
Basic RAG retrieves a few chunks and provides incomplete answers.
Deep Search iteratively explores multiple sources, identifies gaps,
and synthesizes comprehensive reports with citations.

Key Features:
- Multi-source retrieval (web, APIs, knowledge bases)
- Iterative reasoning loop
- Budget management (time/cost)
- Gap identification and follow-up generation
- Citation tracking

Usage:
    python example.py
"""

import os
import sys
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# DATA MODELS
# =============================================================================

class SourceType(Enum):
    """Types of information sources."""
    WEB_SEARCH = "web_search"
    NEWS_API = "news_api"
    FINANCIAL_API = "financial_api"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass
class Source:
    """Represents an information source with citation details."""
    id: str
    title: str
    url: str
    source_type: SourceType
    content: str
    retrieved_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0


@dataclass
class SearchResult:
    """Result from a single search operation."""
    query: str
    sources: List[Source]
    synthesis: str
    confidence: float


@dataclass
class ResearchSection:
    """A section of the research report."""
    query: str
    answer: str
    sources: List[Source]
    subsections: List['ResearchSection'] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Budget:
    """Budget constraints for deep search."""
    max_iterations: int = 5
    max_time_seconds: float = 60.0
    max_cost_dollars: float = 1.0
    
    # Tracking
    iterations_used: int = 0
    time_used: float = 0.0
    cost_used: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def is_exhausted(self) -> Tuple[bool, str]:
        """Check if budget is exhausted."""
        self.time_used = time.time() - self.start_time
        
        if self.iterations_used >= self.max_iterations:
            return True, "max_iterations"
        if self.time_used >= self.max_time_seconds:
            return True, "max_time"
        if self.cost_used >= self.max_cost_dollars:
            return True, "max_cost"
        return False, ""
    
    def add_cost(self, cost: float):
        """Add cost to budget tracking."""
        self.cost_used += cost
    
    def increment_iteration(self):
        """Increment iteration count."""
        self.iterations_used += 1


@dataclass
class DeepSearchResult:
    """Final result from deep search."""
    query: str
    answer: str
    sections: List[ResearchSection]
    all_sources: List[Source]
    budget_summary: Dict[str, Any]
    confidence: float


# =============================================================================
# MULTI-SOURCE RETRIEVER
# =============================================================================

class MultiSourceRetriever:
    """
    Retrieves information from multiple sources.
    
    In production, this would integrate with:
    - Web search APIs (Google, Bing, DuckDuckGo)
    - News APIs (NewsAPI, Bloomberg, Reuters)
    - Financial APIs (Alpha Vantage, Yahoo Finance)
    - Internal knowledge bases
    """
    
    def __init__(self):
        """Initialize with simulated knowledge bases."""
        self._init_simulated_data()
    
    def _init_simulated_data(self):
        """Initialize simulated data for demonstration."""
        # Simulated company data for "TechCorp Inc."
        self.company_data = {
            "techcorp": {
                "overview": {
                    "name": "TechCorp Inc.",
                    "sector": "Technology",
                    "industry": "Cloud Computing & AI",
                    "founded": 2010,
                    "headquarters": "San Francisco, CA",
                    "employees": 15000,
                    "description": "TechCorp is a leading provider of cloud infrastructure and AI services."
                },
                "financials": {
                    "revenue_2023": "$8.5B",
                    "revenue_growth": "32%",
                    "gross_margin": "72%",
                    "operating_margin": "28%",
                    "net_income": "$1.2B",
                    "free_cash_flow": "$2.1B",
                    "debt_to_equity": 0.45,
                    "pe_ratio": 45.2
                },
                "products": [
                    {
                        "name": "TechCloud Platform",
                        "description": "Enterprise cloud infrastructure",
                        "revenue_share": "60%",
                        "growth": "25%"
                    },
                    {
                        "name": "TechAI Suite",
                        "description": "AI/ML platform for enterprises",
                        "revenue_share": "30%",
                        "growth": "85%"
                    },
                    {
                        "name": "TechSecure",
                        "description": "Cloud security services",
                        "revenue_share": "10%",
                        "growth": "40%"
                    }
                ],
                "competitors": ["CloudGiant Corp", "DataScale Inc", "AIFirst Technologies"],
                "recent_news": [
                    {
                        "date": "2024-01-15",
                        "headline": "TechCorp announces $500M investment in AI infrastructure",
                        "summary": "Company plans to expand GPU clusters for AI training."
                    },
                    {
                        "date": "2024-01-10",
                        "headline": "TechCorp partners with major healthcare provider",
                        "summary": "New partnership to bring AI diagnostics to 500 hospitals."
                    },
                    {
                        "date": "2024-01-05",
                        "headline": "TechCorp Q4 earnings beat expectations",
                        "summary": "Revenue up 35% YoY, guidance raised for 2024."
                    }
                ],
                "risks": [
                    "High customer concentration (top 10 customers = 40% revenue)",
                    "Competitive pressure from hyperscalers",
                    "Regulatory risks in AI/data privacy",
                    "Talent acquisition challenges"
                ],
                "opportunities": [
                    "Growing enterprise AI adoption",
                    "International expansion potential",
                    "New product launches in 2024",
                    "Strategic acquisition targets identified"
                ]
            }
        }
        
        # Simulated industry data
        self.industry_data = {
            "cloud_computing": {
                "market_size_2023": "$600B",
                "growth_rate": "18%",
                "key_trends": [
                    "Multi-cloud adoption accelerating",
                    "Edge computing integration",
                    "AI/ML workload migration to cloud",
                    "Sustainability focus in data centers"
                ],
                "major_players": ["AWS", "Azure", "Google Cloud", "TechCorp", "Others"]
            }
        }
    
    def search_web(self, query: str) -> List[Source]:
        """
        Simulate web search results.
        
        In production: Use Google Search API, Bing API, or similar.
        """
        sources = []
        query_lower = query.lower()
        
        # Simulate relevant search results based on query
        if "techcorp" in query_lower or "company" in query_lower:
            sources.append(Source(
                id=f"web_{len(sources)+1}",
                title="TechCorp Inc. - Company Overview",
                url="https://example.com/techcorp-overview",
                source_type=SourceType.WEB_SEARCH,
                content=f"TechCorp Inc. is a leading cloud computing and AI company. "
                        f"Founded in 2010, the company has grown to {self.company_data['techcorp']['overview']['employees']} employees. "
                        f"Their main products include {', '.join([p['name'] for p in self.company_data['techcorp']['products']])}.",
                relevance_score=0.95
            ))
        
        if "financial" in query_lower or "revenue" in query_lower or "investment" in query_lower:
            fin = self.company_data['techcorp']['financials']
            sources.append(Source(
                id=f"web_{len(sources)+1}",
                title="TechCorp Financial Analysis 2024",
                url="https://example.com/techcorp-financials",
                source_type=SourceType.WEB_SEARCH,
                content=f"TechCorp reported revenue of {fin['revenue_2023']} with {fin['revenue_growth']} growth. "
                        f"Gross margin stands at {fin['gross_margin']}, operating margin at {fin['operating_margin']}. "
                        f"P/E ratio is {fin['pe_ratio']}.",
                relevance_score=0.92
            ))
        
        if "competitor" in query_lower or "competition" in query_lower:
            sources.append(Source(
                id=f"web_{len(sources)+1}",
                title="Cloud Computing Market Competition Analysis",
                url="https://example.com/cloud-competition",
                source_type=SourceType.WEB_SEARCH,
                content=f"TechCorp faces competition from {', '.join(self.company_data['techcorp']['competitors'])}. "
                        f"The cloud computing market is valued at {self.industry_data['cloud_computing']['market_size_2023']} "
                        f"with {self.industry_data['cloud_computing']['growth_rate']} annual growth.",
                relevance_score=0.88
            ))
        
        if "risk" in query_lower:
            sources.append(Source(
                id=f"web_{len(sources)+1}",
                title="TechCorp Risk Assessment",
                url="https://example.com/techcorp-risks",
                source_type=SourceType.WEB_SEARCH,
                content=f"Key risks for TechCorp include: {'; '.join(self.company_data['techcorp']['risks'])}.",
                relevance_score=0.85
            ))
        
        if "opportunity" in query_lower or "growth" in query_lower:
            sources.append(Source(
                id=f"web_{len(sources)+1}",
                title="TechCorp Growth Opportunities",
                url="https://example.com/techcorp-opportunities",
                source_type=SourceType.WEB_SEARCH,
                content=f"Growth opportunities for TechCorp: {'; '.join(self.company_data['techcorp']['opportunities'])}.",
                relevance_score=0.87
            ))
        
        # Add general fallback if no specific matches
        if not sources:
            sources.append(Source(
                id=f"web_{len(sources)+1}",
                title="General Market Analysis",
                url="https://example.com/market-analysis",
                source_type=SourceType.WEB_SEARCH,
                content="The technology sector continues to show strong growth driven by AI and cloud adoption.",
                relevance_score=0.60
            ))
        
        return sources
    
    def search_news_api(self, query: str) -> List[Source]:
        """
        Simulate news API results.
        
        In production: Use NewsAPI, Bloomberg API, or similar.
        """
        sources = []
        query_lower = query.lower()
        
        if "techcorp" in query_lower or "news" in query_lower or "recent" in query_lower:
            for news in self.company_data['techcorp']['recent_news']:
                sources.append(Source(
                    id=f"news_{len(sources)+1}",
                    title=news['headline'],
                    url=f"https://news.example.com/{news['date']}",
                    source_type=SourceType.NEWS_API,
                    content=f"{news['date']}: {news['headline']}. {news['summary']}",
                    relevance_score=0.90
                ))
        
        return sources
    
    def search_financial_api(self, query: str) -> List[Source]:
        """
        Simulate financial API results.
        
        In production: Use Alpha Vantage, Yahoo Finance, Bloomberg API.
        """
        sources = []
        query_lower = query.lower()
        
        if "financial" in query_lower or "revenue" in query_lower or "earnings" in query_lower or "valuation" in query_lower:
            fin = self.company_data['techcorp']['financials']
            sources.append(Source(
                id=f"financial_{len(sources)+1}",
                title="TechCorp Inc. (TECH) - Financial Data",
                url="https://finance.example.com/TECH",
                source_type=SourceType.FINANCIAL_API,
                content=f"Financial Metrics: Revenue: {fin['revenue_2023']}, "
                        f"Revenue Growth: {fin['revenue_growth']}, "
                        f"Net Income: {fin['net_income']}, "
                        f"Free Cash Flow: {fin['free_cash_flow']}, "
                        f"P/E Ratio: {fin['pe_ratio']}, "
                        f"Debt/Equity: {fin['debt_to_equity']}",
                relevance_score=0.98
            ))
        
        return sources
    
    def search_knowledge_base(self, query: str) -> List[Source]:
        """
        Simulate internal knowledge base search.
        
        In production: Use vector database like Pinecone, Weaviate, or Milvus.
        """
        sources = []
        query_lower = query.lower()
        
        if "product" in query_lower:
            for product in self.company_data['techcorp']['products']:
                sources.append(Source(
                    id=f"kb_{len(sources)+1}",
                    title=f"Product Analysis: {product['name']}",
                    url=f"https://kb.internal/products/{product['name'].lower().replace(' ', '-')}",
                    source_type=SourceType.KNOWLEDGE_BASE,
                    content=f"{product['name']}: {product['description']}. "
                            f"Revenue share: {product['revenue_share']}, Growth: {product['growth']}.",
                    relevance_score=0.93
                ))
        
        if "industry" in query_lower or "market" in query_lower:
            ind = self.industry_data['cloud_computing']
            sources.append(Source(
                id=f"kb_{len(sources)+1}",
                title="Cloud Computing Industry Analysis",
                url="https://kb.internal/industries/cloud-computing",
                source_type=SourceType.KNOWLEDGE_BASE,
                content=f"Cloud Computing Market: Size {ind['market_size_2023']}, "
                        f"Growth {ind['growth_rate']}. "
                        f"Key trends: {'; '.join(ind['key_trends'])}.",
                relevance_score=0.91
            ))
        
        return sources
    
    def retrieve(self, query: str) -> List[Source]:
        """
        Retrieve from all sources and combine results.
        
        This is the main entry point for multi-source retrieval.
        """
        all_sources = []
        
        # Search all sources
        all_sources.extend(self.search_web(query))
        all_sources.extend(self.search_news_api(query))
        all_sources.extend(self.search_financial_api(query))
        all_sources.extend(self.search_knowledge_base(query))
        
        # Sort by relevance
        all_sources.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return all_sources


# =============================================================================
# LLM REASONER
# =============================================================================

class LLMReasoner:
    """
    Handles reasoning tasks: synthesis, gap identification, follow-up generation.
    
    In production, this would use an actual LLM (Ollama, OpenAI, etc.).
    """
    
    def __init__(self, use_ollama: bool = False):
        """
        Initialize the reasoner.
        
        Args:
            use_ollama: If True, use Ollama for actual LLM calls (not implemented in demo)
        """
        self.use_ollama = use_ollama
    
    def synthesize(self, query: str, sources: List[Source]) -> Tuple[str, float]:
        """
        Synthesize an answer from retrieved sources.
        
        Returns:
            Tuple of (synthesized_answer, confidence_score)
        """
        if not sources:
            return "No relevant information found.", 0.0
        
        # Combine source content
        combined_content = "\n".join([
            f"[{s.source_type.value}] {s.content}" for s in sources
        ])
        
        # Simulated synthesis (in production, use LLM)
        # The synthesis combines key information from sources
        synthesis_parts = []
        confidence = 0.0
        
        for source in sources[:5]:  # Use top 5 sources
            if source.relevance_score > 0.7:
                synthesis_parts.append(source.content)
                confidence += source.relevance_score * 0.2
        
        if synthesis_parts:
            synthesis = " ".join(synthesis_parts)
            # Truncate if too long
            if len(synthesis) > 1000:
                synthesis = synthesis[:1000] + "..."
        else:
            synthesis = "Limited information available for this query."
            confidence = 0.3
        
        confidence = min(confidence, 1.0)
        
        return synthesis, confidence
    
    def identify_gaps(self, query: str, current_answer: str, sources: List[Source]) -> List[str]:
        """
        Identify information gaps in the current answer.
        
        Returns:
            List of identified gaps/missing information areas
        """
        # Simulated gap identification based on query analysis
        gaps = []
        query_lower = query.lower()
        answer_lower = current_answer.lower()
        
        # Investment research always needs these topics - check coverage depth
        essential_topics = {
            "financial": {
                "desc": "Detailed financial metrics and valuation analysis",
                "depth_keywords": ["pe ratio", "margin", "cash flow", "debt", "valuation"]
            },
            "competitor": {
                "desc": "Competitive landscape and market positioning",
                "depth_keywords": ["market share", "competitive advantage", "comparison", "vs"]
            },
            "risk": {
                "desc": "Risk factors and mitigation strategies",
                "depth_keywords": ["risk", "threat", "challenge", "vulnerability"]
            },
            "growth": {
                "desc": "Growth opportunities and future outlook",
                "depth_keywords": ["opportunity", "expansion", "growth", "forecast"]
            },
            "product": {
                "desc": "Product portfolio and revenue breakdown",
                "depth_keywords": ["product", "service", "offering", "portfolio"]
            }
        }
        
        # For investment queries, always explore key areas
        if "invest" in query_lower or "evaluat" in query_lower or "factor" in query_lower:
            for topic, info in essential_topics.items():
                # Check if topic is deeply covered (has depth keywords)
                depth_covered = sum(1 for kw in info["depth_keywords"] if kw in answer_lower)
                if depth_covered < 2:  # Not deeply covered
                    gaps.append(info["desc"])
        
        # For product queries, check competitive analysis
        if "product" in query_lower or "competitive" in query_lower:
            if "competitor" not in answer_lower and "market share" not in answer_lower:
                gaps.append("Competitive landscape and market positioning")
        
        # Always check for recent developments
        if "recent" not in answer_lower or "news" not in answer_lower:
            if len(gaps) < 3:
                gaps.append("Recent news and developments")
        
        # Limit to top 3 gaps
        return gaps[:3]
    
    def generate_follow_ups(self, original_query: str, gaps: List[str]) -> List[str]:
        """
        Generate follow-up queries to fill identified gaps.
        
        Returns:
            List of follow-up queries
        """
        follow_ups = []
        
        # Extract company name from query if present
        company_name = "TechCorp"  # Default for demo
        if "techcorp" in original_query.lower():
            company_name = "TechCorp"
        
        for gap in gaps:
            # Generate specific follow-up query for each gap
            if "financial" in gap.lower():
                follow_ups.append(f"What are the detailed financial metrics for {company_name}?")
            elif "competitor" in gap.lower():
                follow_ups.append(f"Who are {company_name}'s main competitors and how do they compare?")
            elif "risk" in gap.lower():
                follow_ups.append(f"What are the key risk factors for investing in {company_name}?")
            elif "growth" in gap.lower():
                follow_ups.append(f"What are the growth opportunities for {company_name}?")
            elif "product" in gap.lower():
                follow_ups.append(f"What products does {company_name} offer and what is their revenue breakdown?")
            elif "management" in gap.lower():
                follow_ups.append(f"Who is on {company_name}'s management team?")
            elif "recent" in gap.lower() or "news" in gap.lower():
                follow_ups.append(f"What are the recent news and developments for {company_name}?")
            else:
                follow_ups.append(f"Tell me more about {gap} for {company_name}")
        
        return follow_ups
    
    def assess_answer_quality(self, query: str, answer: str, sections: List[ResearchSection]) -> Tuple[bool, float]:
        """
        Assess if the answer is comprehensive enough.
        
        Returns:
            Tuple of (is_good_enough, quality_score)
        """
        # Calculate quality based on coverage and confidence
        if not sections:
            return False, 0.0
        
        # Average confidence across sections
        avg_confidence = sum(s.confidence for s in sections) / len(sections)
        
        # Check coverage of key topics
        answer_lower = answer.lower()
        key_topics = ["financial", "product", "competitor", "risk", "growth"]
        topics_covered = sum(1 for topic in key_topics if topic in answer_lower)
        coverage_score = topics_covered / len(key_topics)
        
        # Combined quality score
        quality_score = (avg_confidence + coverage_score) / 2
        
        # Consider answer good enough if quality > 0.7 and at least 3 sections
        is_good_enough = quality_score > 0.7 and len(sections) >= 3
        
        return is_good_enough, quality_score
    
    def final_synthesis(self, query: str, sections: List[ResearchSection]) -> str:
        """
        Create final comprehensive answer from all sections.
        """
        if not sections:
            return "Unable to generate a comprehensive answer."
        
        # Build structured answer
        answer_parts = [
            f"## Research Report: {query}",
            "",
            "### Executive Summary",
            ""
        ]
        
        # Add executive summary from first section
        if sections:
            answer_parts.append(sections[0].answer[:500])
            answer_parts.append("")
        
        # Add detailed sections
        for i, section in enumerate(sections, 1):
            answer_parts.append(f"### {i}. {section.query}")
            answer_parts.append("")
            answer_parts.append(section.answer)
            answer_parts.append("")
            
            # Add citations for this section
            if section.sources:
                answer_parts.append("**Sources:**")
                for j, source in enumerate(section.sources[:3], 1):
                    answer_parts.append(f"  [{j}] {source.title} ({source.url})")
                answer_parts.append("")
        
        return "\n".join(answer_parts)


# =============================================================================
# DEEP SEARCH ORCHESTRATOR
# =============================================================================

class DeepSearchOrchestrator:
    """
    Orchestrates the deep search process.
    
    This is the main class that coordinates:
    - Multi-source retrieval
    - Iterative reasoning
    - Budget management
    - Answer synthesis
    """
    
    def __init__(self, budget: Optional[Budget] = None):
        """
        Initialize the orchestrator.
        
        Args:
            budget: Budget constraints for the search
        """
        self.retriever = MultiSourceRetriever()
        self.reasoner = LLMReasoner()
        self.budget = budget or Budget()
        self.all_sources: List[Source] = []
        self.sections: List[ResearchSection] = []
    
    def _create_section(self, query: str) -> ResearchSection:
        """
        Create a research section for a query.
        
        STEP 1 of Deep Search: Retrieve and synthesize
        """
        # Retrieve from multiple sources
        sources = self.retriever.retrieve(query)
        self.all_sources.extend(sources)
        
        # Track cost (simulated)
        self.budget.add_cost(0.01 * len(sources))  # $0.01 per source
        
        # Synthesize answer
        answer, confidence = self.reasoner.synthesize(query, sources)
        
        return ResearchSection(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    
    def _add_subsections(self, parent: ResearchSection):
        """
        Add subsections by identifying gaps and generating follow-ups.
        
        STEP 2 of Deep Search: Identify gaps and follow up
        """
        # Identify gaps in current answer
        gaps = self.reasoner.identify_gaps(parent.query, parent.answer, parent.sources)
        
        if not gaps:
            return
        
        # Generate follow-up queries
        follow_ups = self.reasoner.generate_follow_ups(parent.query, gaps)
        
        # Create subsections for each follow-up
        for follow_up in follow_ups:
            # Check budget before each subsection
            exhausted, reason = self.budget.is_exhausted()
            if exhausted:
                break
            
            subsection = self._create_section(follow_up)
            parent.subsections.append(subsection)
            self.budget.add_cost(0.02)  # LLM cost
    
    def search(self, query: str, depth: int = 2) -> DeepSearchResult:
        """
        Execute deep search for a query.
        
        MAIN LOOP: Iterative retrieval and reasoning
        
        Args:
            query: The main research question
            depth: How deep to go with follow-ups (default 2)
        
        Returns:
            DeepSearchResult with comprehensive answer
        """
        print(f"\n{'='*60}")
        print(f"DEEP SEARCH: {query}")
        print(f"{'='*60}")
        
        # STEP 1: Initial retrieval and synthesis
        print(f"\n📥 Iteration 1: Initial retrieval...")
        self.budget.increment_iteration()
        root_section = self._create_section(query)
        self.sections.append(root_section)
        print(f"   Retrieved {len(root_section.sources)} sources, confidence: {root_section.confidence:.2f}")
        
        # STEP 2: Iterative deep search
        current_depth = 0
        sections_to_expand = [root_section]
        
        while current_depth < depth:
            current_depth += 1
            
            # Check budget
            exhausted, reason = self.budget.is_exhausted()
            if exhausted:
                print(f"\n⚠️  Budget exhausted: {reason}")
                break
            
            print(f"\n🔄 Depth {current_depth}: Expanding {len(sections_to_expand)} sections...")
            
            next_sections = []
            for section in sections_to_expand:
                self.budget.increment_iteration()
                
                # Check budget before each expansion
                exhausted, reason = self.budget.is_exhausted()
                if exhausted:
                    print(f"\n⚠️  Budget exhausted: {reason}")
                    break
                
                # Add subsections (identify gaps, follow up)
                self._add_subsections(section)
                
                for subsection in section.subsections:
                    self.sections.append(subsection)
                    next_sections.append(subsection)
                    print(f"   + {subsection.query[:50]}... (conf: {subsection.confidence:.2f})")
            
            sections_to_expand = next_sections
            
            # Check if answer is good enough
            is_good_enough, quality = self.reasoner.assess_answer_quality(
                query, root_section.answer, self.sections
            )
            if is_good_enough:
                print(f"\n✅ Answer quality sufficient: {quality:.2f}")
                break
        
        # STEP 3: Final synthesis
        print(f"\n📝 Final synthesis...")
        final_answer = self.reasoner.final_synthesis(query, self.sections)
        
        # Calculate overall confidence
        overall_confidence = sum(s.confidence for s in self.sections) / len(self.sections) if self.sections else 0.0
        
        # Build result
        result = DeepSearchResult(
            query=query,
            answer=final_answer,
            sections=self.sections,
            all_sources=self.all_sources,
            budget_summary={
                "iterations_used": self.budget.iterations_used,
                "time_used": round(time.time() - self.budget.start_time, 2),
                "cost_used": round(self.budget.cost_used, 4),
                "sections_created": len(self.sections),
                "total_sources": len(self.all_sources)
            },
            confidence=overall_confidence
        )
        
        return result


# =============================================================================
# CITATION MANAGER
# =============================================================================

class CitationManager:
    """Manages citations for the research report."""
    
    @staticmethod
    def format_citations(sources: List[Source]) -> str:
        """Format sources as numbered citations."""
        if not sources:
            return "No sources available."
        
        # Deduplicate by URL
        seen_urls = set()
        unique_sources = []
        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        # Format as numbered list
        citations = ["## References", ""]
        for i, source in enumerate(unique_sources, 1):
            citations.append(f"[{i}] **{source.title}**")
            citations.append(f"    URL: {source.url}")
            citations.append(f"    Source: {source.source_type.value}")
            citations.append(f"    Retrieved: {source.retrieved_at.strftime('%Y-%m-%d %H:%M')}")
            citations.append("")
        
        return "\n".join(citations)


# =============================================================================
# MARKET RESEARCH ANALYST
# =============================================================================

class MarketResearchAnalyst:
    """
    Market Research Analyst System using Deep Search.
    
    This is the main entry point for conducting comprehensive market research.
    """
    
    def __init__(self):
        """Initialize the analyst."""
        self.citation_manager = CitationManager()
    
    def research(self, query: str, max_iterations: int = 5, 
                 max_time_seconds: float = 60.0) -> DeepSearchResult:
        """
        Conduct comprehensive market research.
        
        Args:
            query: The research question
            max_iterations: Maximum number of search iterations
            max_time_seconds: Maximum time budget in seconds
        
        Returns:
            DeepSearchResult with comprehensive research report
        """
        # Create budget
        budget = Budget(
            max_iterations=max_iterations,
            max_time_seconds=max_time_seconds,
            max_cost_dollars=1.0
        )
        
        # Create orchestrator and run search
        orchestrator = DeepSearchOrchestrator(budget)
        result = orchestrator.search(query, depth=2)
        
        return result
    
    def format_report(self, result: DeepSearchResult) -> str:
        """Format the result as a readable report."""
        report_parts = [
            "=" * 70,
            "MARKET RESEARCH REPORT",
            "=" * 70,
            "",
            result.answer,
            "",
            "-" * 70,
            "",
            self.citation_manager.format_citations(result.all_sources),
            "",
            "-" * 70,
            "SEARCH METRICS",
            "-" * 70,
            f"Query: {result.query}",
            f"Overall Confidence: {result.confidence:.2%}",
            f"Iterations: {result.budget_summary['iterations_used']}",
            f"Time Used: {result.budget_summary['time_used']}s",
            f"Estimated Cost: ${result.budget_summary['cost_used']:.4f}",
            f"Sections Created: {result.budget_summary['sections_created']}",
            f"Total Sources: {result.budget_summary['total_sources']}",
            "=" * 70
        ]
        
        return "\n".join(report_parts)


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Main demonstration of the Deep Search pattern."""
    print("=" * 70)
    print("DEEP SEARCH PATTERN - MARKET RESEARCH ANALYST")
    print("=" * 70)
    print()
    print("This example demonstrates Deep Search for comprehensive market research.")
    print("The system iteratively retrieves, analyzes, and synthesizes information")
    print("until a comprehensive answer is found or budget is exhausted.")
    print()
    
    # Create analyst
    analyst = MarketResearchAnalyst()
    
    # Research query
    query = "What factors should I consider when evaluating TechCorp as an investment?"
    
    print(f"Research Query: {query}")
    print()
    
    # Conduct research
    result = analyst.research(
        query=query,
        max_iterations=10,
        max_time_seconds=30.0
    )
    
    # Format and print report
    report = analyst.format_report(result)
    print()
    print(report)
    
    # Additional demonstration: Different query
    print("\n" + "=" * 70)
    print("SECOND RESEARCH QUERY")
    print("=" * 70)
    
    query2 = "What are TechCorp's main products and their competitive position?"
    print(f"\nResearch Query: {query2}")
    
    result2 = analyst.research(
        query=query2,
        max_iterations=5,
        max_time_seconds=20.0
    )
    
    report2 = analyst.format_report(result2)
    print()
    print(report2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
