"""
Reverse Neutralization Pattern

This example demonstrates the reverse neutralization approach:

STEP 1: Generate Neutral Form
    - Generate content in neutral, standardized format
    - Easier for model (no style knowledge needed)

STEP 2: Create Training Dataset
    - Collect pairs: neutral form → desired style
    - Manual or semi-automated collection

STEP 3: Fine-Tune the Model
    - Train model on neutral → style pairs
    - Learn the style transformation

STEP 4: Inference
    - Generate neutral content
    - Convert to desired style using fine-tuned model

USE CASE: Technical Documentation to Personal Blog Style
    Realistic scenario: Converting technical documentation into
    your personal blog writing style for a technical blog.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class StylePair:
    """Represents a neutral → style pair for training."""
    neutral: str
    styled: str
    description: Optional[str] = None


# ============================================================================
# STEP 1: Generate Neutral Form
# ============================================================================

class NeutralGenerator:
    """
    STEP 1: Generate Neutral Form
    
    Generates content in neutral, standardized format.
    This is easier because no personal style knowledge is needed.
    """
    
    def generate_neutral(self, topic: str, content_type: str = "documentation") -> str:
        """
        Generate neutral form content.
        
        In production, this would use a base LLM to generate
        neutral, standardized content.
        """
        # Simulate neutral generation
        if content_type == "documentation":
            return f"""# {topic}

## Overview
This section provides a comprehensive overview of {topic}. The following information covers key concepts, implementation details, and best practices.

## Key Concepts
The primary concepts related to {topic} include fundamental principles and core functionality. Understanding these concepts is essential for effective implementation.

## Implementation
To implement {topic}, follow these steps:
1. Initialize the required components
2. Configure the necessary parameters
3. Execute the implementation process
4. Verify the results

## Best Practices
When working with {topic}, consider the following best practices:
- Ensure proper configuration
- Follow established patterns
- Validate inputs and outputs
- Handle errors appropriately

## Conclusion
This documentation provides a foundation for understanding and implementing {topic}. For additional information, refer to the official documentation."""
        
        return f"Neutral content about {topic}"
    
    def generate_multiple_neutral(self, topics: List[str]) -> List[str]:
        """Generate neutral content for multiple topics."""
        return [self.generate_neutral(topic) for topic in topics]


# ============================================================================
# STEP 2: Create Training Dataset
# ============================================================================

class TrainingDataCollector:
    """
    STEP 2: Create Training Dataset
    
    Collects pairs of neutral form → desired style for fine-tuning.
    """
    
    def __init__(self):
        self.pairs: List[StylePair] = []
    
    def add_pair(self, neutral: str, styled: str, description: str = ""):
        """Add a neutral → style pair to the dataset."""
        self.pairs.append(StylePair(neutral, styled, description))
    
    def create_training_data(self) -> List[Dict[str, str]]:
        """
        Format training data for fine-tuning.
        
        Format: List of prompt-completion pairs
        """
        training_data = []
        
        for pair in self.pairs:
            training_data.append({
                "prompt": f"Convert this technical documentation to personal blog style:\n\n{pair.neutral}\n\nBlog post:",
                "completion": pair.styled
            })
        
        return training_data
    
    def save_training_data(self, filepath: str):
        """Save training data to JSONL file for fine-tuning."""
        training_data = self.create_training_data()
        
        with open(filepath, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved {len(training_data)} training examples to {filepath}")


# ============================================================================
# STEP 3: Fine-Tune the Model
# ============================================================================

class ModelFineTuner:
    """
    STEP 3: Fine-Tune the Model
    
    Fine-tunes a base model on neutral → style pairs.
    """
    
    def __init__(self, base_model: str = "gpt-3.5-turbo"):
        self.base_model = base_model
        self.fine_tuned_model_id = None
    
    def prepare_fine_tuning(self, training_file: str):
        """
        Prepare fine-tuning job.
        
        In production, this would:
        1. Upload training file to fine-tuning service
        2. Create fine-tuning job
        3. Monitor training progress
        4. Get fine-tuned model ID
        """
        logger.info(f"Preparing fine-tuning job for {self.base_model}")
        logger.info(f"Training file: {training_file}")
        
        # In production:
        # 1. Upload file: client.files.create(file=open(training_file, "rb"), purpose="fine-tune")
        # 2. Create job: client.fine_tuning.jobs.create(training_file=file.id, model=base_model)
        # 3. Monitor: client.fine_tuning.jobs.retrieve(job_id)
        # 4. Get model: job.fine_tuned_model
        
        logger.info("Fine-tuning would proceed here...")
        logger.info("After completion, fine-tuned model ID would be available")
        
        # Simulate fine-tuned model ID
        self.fine_tuned_model_id = f"ft-{self.base_model}-personal-style"
    
    def validate_fine_tuning(self, validation_examples: List[StylePair]) -> Dict[str, Any]:
        """
        Validate fine-tuned model on held-out examples.
        
        Returns validation metrics.
        """
        logger.info(f"Validating fine-tuned model on {len(validation_examples)} examples")
        
        # In production, would test fine-tuned model on validation set
        # and compute metrics like style consistency, content preservation, etc.
        
        return {
            "style_consistency": 0.92,
            "content_preservation": 0.95,
            "overall_quality": 0.93
        }


# ============================================================================
# STEP 4: Inference
# ============================================================================

class StyleConverter:
    """
    STEP 4: Inference
    
    Uses fine-tuned model to convert neutral content to desired style.
    """
    
    def __init__(self, fine_tuned_model_id: str):
        self.fine_tuned_model_id = fine_tuned_model_id
    
    def convert_to_style(self, neutral_content: str) -> str:
        """
        Convert neutral content to desired style using fine-tuned model.
        
        In production, this would call the fine-tuned model.
        """
        # In production:
        # response = fine_tuned_model.generate(
        #     prompt=f"Convert this technical documentation to personal blog style:\n\n{neutral_content}\n\nBlog post:"
        # )
        # return response
        
        # Simulate conversion
        logger.info("Using fine-tuned model to convert neutral to personal style")
        
        # Simple simulation - in production, this would be the actual model output
        return f"[Fine-tuned model output: {neutral_content[:50]}...]"
    
    def generate_in_style(self, topic: str, neutral_generator: NeutralGenerator) -> str:
        """
        Complete pipeline: Generate neutral, then convert to style.
        
        This is the full reverse neutralization workflow.
        """
        # Step 1: Generate neutral form
        neutral = neutral_generator.generate_neutral(topic)
        
        # Step 4: Convert to style using fine-tuned model
        styled = self.convert_to_style(neutral)
        
        return styled


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_step_1():
    """Demonstrate Step 1: Generate Neutral Form."""
    print("\n" + "="*70)
    print("STEP 1: Generate Neutral Form")
    print("="*70)
    
    print("\n💡 Why Neutral Form First?")
    print("   • Easier for model (no style knowledge needed)")
    print("   • More consistent and reliable")
    print("   • Standard formats are well-learned")
    print("   • Separates content from style")
    
    generator = NeutralGenerator()
    
    print("\n📝 Example: Generating Neutral Technical Documentation")
    topic = "API Authentication"
    neutral = generator.generate_neutral(topic)
    
    print(f"\n   Topic: {topic}")
    print(f"\n   Neutral Form (first 300 chars):")
    print(f"   {neutral[:300]}...")
    
    print("\n✅ Neutral form generated - ready for style conversion")


def demonstrate_step_2():
    """Demonstrate Step 2: Create Training Dataset."""
    print("\n" + "="*70)
    print("STEP 2: Create Training Dataset")
    print("="*70)
    
    print("\n📊 Collecting Neutral → Style Pairs")
    print("   Process:")
    print("   1. Generate neutral content (Step 1)")
    print("   2. Rewrite in your personal style")
    print("   3. Create pairs for training")
    
    collector = TrainingDataCollector()
    
    # Example pair
    neutral = """# API Authentication

## Overview
This section provides a comprehensive overview of API authentication. The following information covers key concepts, implementation details, and best practices.

## Key Concepts
The primary concepts related to API authentication include fundamental principles and core functionality."""
    
    styled = """# How I Learned to Love API Authentication

Let me tell you about API authentication - it's one of those things that seems scary at first, but once you get it, it's actually pretty straightforward.

I remember when I first started working with APIs. I was confused about all the different authentication methods. But here's the thing: they all boil down to proving who you are. Think of it like showing your ID at a bar - the API needs to know you're allowed to be there.

The main concepts you need to understand are pretty simple once you break them down."""
    
    collector.add_pair(
        neutral,
        styled,
        "Technical doc to personal blog style"
    )
    
    print("\n   Example Pair:")
    print(f"   Neutral (first 100 chars): {neutral[:100]}...")
    print(f"   Styled (first 100 chars): {styled[:100]}...")
    
    training_data = collector.create_training_data()
    print(f"\n   Created {len(training_data)} training examples")
    print("\n   Training data format:")
    print(json.dumps(training_data[0], indent=2)[:500] + "...")
    
    print("\n✅ Training dataset created - ready for fine-tuning")
    print("   (In production, collect 100+ high-quality pairs)")


def demonstrate_step_3():
    """Demonstrate Step 3: Fine-Tune the Model."""
    print("\n" + "="*70)
    print("STEP 3: Fine-Tune the Model")
    print("="*70)
    
    print("\n🔧 Fine-Tuning Process:")
    print("   1. Format training data (JSONL)")
    print("   2. Upload to fine-tuning service")
    print("   3. Create fine-tuning job")
    print("   4. Monitor training progress")
    print("   5. Get fine-tuned model ID")
    
    tuner = ModelFineTuner()
    
    # Simulate fine-tuning
    print("\n   Preparing fine-tuning job...")
    tuner.prepare_fine_tuning("training_data.jsonl")
    
    print(f"\n   Fine-tuned model ID: {tuner.fine_tuned_model_id}")
    
    # Validate
    validation_examples = [
        StylePair("Neutral content 1", "Styled content 1"),
        StylePair("Neutral content 2", "Styled content 2")
    ]
    
    metrics = tuner.validate_fine_tuning(validation_examples)
    print("\n   Validation Metrics:")
    print(f"   • Style Consistency: {metrics['style_consistency']:.2%}")
    print(f"   • Content Preservation: {metrics['content_preservation']:.2%}")
    print(f"   • Overall Quality: {metrics['overall_quality']:.2%}")
    
    print("\n✅ Model fine-tuned - ready for inference")


def demonstrate_step_4():
    """Demonstrate Step 4: Inference."""
    print("\n" + "="*70)
    print("STEP 4: Inference")
    print("="*70)
    
    print("\n🚀 Using Fine-Tuned Model for Style Conversion")
    
    converter = StyleConverter("ft-gpt-3.5-turbo-personal-style")
    generator = NeutralGenerator()
    
    print("\n   Complete Workflow:")
    print("   1. Generate neutral content")
    print("   2. Convert to personal style using fine-tuned model")
    
    topic = "Database Indexing"
    result = converter.generate_in_style(topic, generator)
    
    print(f"\n   Topic: {topic}")
    print(f"   Result: {result}")
    
    print("\n✅ Style conversion complete - content in your personal style")


def show_comparison():
    """Show comparison with zero-shot approach."""
    print("\n" + "="*70)
    print("⚖️  COMPARISON: Zero-Shot vs Reverse Neutralization")
    print("="*70)
    
    print("\n❌ Zero-Shot Approach:")
    print("   Prompt: 'Write about API authentication in my personal blog style'")
    print("   Problems:")
    print("     • Model doesn't know your style")
    print("     • Generic style descriptions don't work")
    print("     • Inconsistent results")
    print("     • Doesn't capture personal nuances")
    
    print("\n✅ Reverse Neutralization:")
    print("   Step 1: Generate neutral technical doc (easy)")
    print("   Step 2-3: Fine-tune model on your style examples")
    print("   Step 4: Convert neutral → your style (consistent)")
    print("   Benefits:")
    print("     • Learns your specific style")
    print("     • Consistent results")
    print("     • Captures personal nuances")
    print("     • Separates content from style")
    
    print("\n💡 Key Insight:")
    print("   Generating neutral content is easier than generating")
    print("   in a specific unknown style. Fine-tuning bridges the gap.")


def show_real_world_example():
    """Show realistic use case example."""
    print("\n" + "="*70)
    print("🌍 REAL-WORLD USE CASE: Technical Blog Writing")
    print("="*70)
    
    print("\nScenario: You run a technical blog and want to convert")
    print("technical documentation into your personal writing style")
    
    print("\n📝 Challenge:")
    print("   • Technical docs are dry and formal")
    print("   • Your blog has a conversational, personal style")
    print("   • Zero-shot doesn't capture your unique voice")
    print("   • Need consistent style across all posts")
    
    print("\n✅ Solution: Reverse Neutralization")
    print("\n   Step 1: Generate neutral technical documentation")
    neutral = """# API Rate Limiting

## Overview
API rate limiting is a mechanism that controls the number of requests a client can make to an API within a specific time period.

## Implementation
Implement rate limiting using token bucket algorithm."""
    
    print(f"   {neutral[:100]}...")
    
    print("\n   Step 2-3: Fine-tune on your blog posts")
    print("   • Collect 100+ examples of your writing")
    print("   • Create neutral → your style pairs")
    print("   • Fine-tune model")
    
    print("\n   Step 4: Convert to your style")
    styled = """# Why API Rate Limiting Matters (And How I Learned the Hard Way)

So, API rate limiting. I'll be honest - I didn't think much about it until I accidentally DDoS'd my own API. Oops.

Here's the thing: rate limiting is basically the API's way of saying 'slow down, buddy.' It controls how many requests you can make in a certain time period. Think of it like a bouncer at a club - they only let so many people in at once.

I use the token bucket algorithm because... well, it works, and I understand it."""
    
    print(f"   {styled[:150]}...")
    
    print("\n" + "="*70)
    print("🎯 Impact: Consistent personal style, faster content creation")
    print("="*70)


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 REVERSE NEUTRALIZATION PATTERN")
    print("="*70)
    
    print("\n📋 Pattern Overview:")
    print("   Two-stage approach: Generate neutral, then convert to style")
    print("   Solves the problem of generating in unknown personal styles")
    print("   Uses fine-tuning to learn style transformation")
    
    # Demonstrate all steps
    demonstrate_step_1()
    demonstrate_step_2()
    demonstrate_step_3()
    demonstrate_step_4()
    
    # Show comparisons and use cases
    show_comparison()
    show_real_world_example()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for detailed explanation")
    print("   2. Collect your writing examples")
    print("   3. Create neutral → style pairs")
    print("   4. Fine-tune model on your style")
    print("   5. Use for consistent style generation")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

