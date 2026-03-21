"""
Content Optimization Pattern

This example demonstrates content optimization using preference-based fine-tuning:

STEP 1: Generate Pair of Contents
    - Generate two variations from same prompt
    - Use different sampling strategies for diversity

STEP 2: Compare and Pick Winner
    - Compare based on optimization goal (e.g., open rates)
    - Select the winning content

STEP 3: Create Training Dataset
    - Collect many preference pairs
    - Format: (prompt, chosen, rejected)

STEP 4: Perform Preference Tuning
    - Use DPO (Direct Preference Optimization) to fine-tune
    - Model learns to generate winning content

STEP 5: Use Tuned LLM
    - Generate optimized content going forward

USE CASE: Email Subject Line Optimization
    Realistic scenario: Optimizing email subject lines to maximize
    open rates for marketing campaigns.
"""

import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ContentPair:
    """Represents a pair of content variations."""
    prompt: str
    variation_a: str
    variation_b: str
    winner: Optional[str] = None  # 'a' or 'b'
    metrics: Optional[Dict[str, float]] = None


@dataclass
class PreferenceExample:
    """Represents a preference for training."""
    prompt: str
    chosen: str
    rejected: str
    reason: Optional[str] = None


# ============================================================================
# STEP 1: Generate Pair of Contents
# ============================================================================

class ContentGenerator:
    """
    STEP 1: Generate Pair of Contents
    
    Generates two different variations from the same prompt.
    Uses different sampling strategies to ensure diversity.
    """
    
    def generate_pair(self, prompt: str, content_type: str = "subject_line") -> Tuple[str, str]:
        """
        Generate two variations from the same prompt.
        
        In production, this would use an LLM with different
        temperature/sampling settings for each variation.
        """
        # Simulate generation with different strategies
        if content_type == "subject_line":
            # Variation A: More direct, action-oriented
            variation_a = self._generate_direct_style(prompt)
            # Variation B: More curiosity-driven, emotional
            variation_b = self._generate_curiosity_style(prompt)
        else:
            variation_a = f"Variation A for: {prompt}"
            variation_b = f"Variation B for: {prompt}"
        
        return variation_a, variation_b
    
    def _generate_direct_style(self, prompt: str) -> str:
        """Generate direct, action-oriented content."""
        # Simulate different generation strategies
        templates = [
            f"New: {prompt}",
            f"Action Required: {prompt}",
            f"Important Update: {prompt}",
            f"Don't Miss: {prompt}",
        ]
        return random.choice(templates)
    
    def _generate_curiosity_style(self, prompt: str) -> str:
        """Generate curiosity-driven, emotional content."""
        templates = [
            f"You won't believe this: {prompt}",
            f"What if I told you about {prompt}?",
            f"The secret to {prompt}",
            f"Why everyone's talking about {prompt}",
        ]
        return random.choice(templates)
    
    def generate_multiple_pairs(self, prompts: List[str]) -> List[ContentPair]:
        """Generate pairs for multiple prompts."""
        pairs = []
        for prompt in prompts:
            var_a, var_b = self.generate_pair(prompt)
            pairs.append(ContentPair(
                prompt=prompt,
                variation_a=var_a,
                variation_b=var_b
            ))
        return pairs


# ============================================================================
# STEP 2: Compare and Pick Winner
# ============================================================================

class ContentComparator:
    """
    STEP 2: Compare and Pick Winner
    
    Compares two content pieces and selects the winner based on
    optimization goal (e.g., open rates, conversions, engagement).
    """
    
    def __init__(self, optimization_goal: str = "open_rate"):
        """
        Initialize comparator.
        
        Args:
            optimization_goal: What to optimize for (open_rate, conversion, engagement)
        """
        self.optimization_goal = optimization_goal
    
    def compare(self, pair: ContentPair, simulate_metrics: bool = True) -> ContentPair:
        """
        Compare two content variations and pick winner.
        
        In production, this would:
        - Send both variations to test audience
        - Measure actual metrics (open rates, clicks, etc.)
        - Select winner based on metrics
        """
        if simulate_metrics:
            # Simulate metrics (in production, these would be real)
            pair.metrics = {
                'variation_a': self._simulate_metric(pair.variation_a),
                'variation_b': self._simulate_metric(pair.variation_b)
            }
        
        # Pick winner based on metrics
        if pair.metrics['variation_a'] > pair.metrics['variation_b']:
            pair.winner = 'a'
        else:
            pair.winner = 'b'
        
        return pair
    
    def _simulate_metric(self, content: str) -> float:
        """Simulate optimization metric (e.g., open rate)."""
        # Simple heuristic: shorter, more direct content often performs better
        # In production, this would be actual measured metrics
        base_score = 0.5
        if len(content) < 50:
            base_score += 0.1
        if '!' in content or '?' in content:
            base_score += 0.05
        return base_score + random.uniform(-0.1, 0.1)
    
    def compare_batch(self, pairs: List[ContentPair]) -> List[ContentPair]:
        """Compare multiple pairs."""
        return [self.compare(pair) for pair in pairs]


# ============================================================================
# STEP 3: Create Training Dataset
# ============================================================================

class PreferenceDatasetBuilder:
    """
    STEP 3: Create Training Dataset
    
    Builds preference dataset from content pairs with winners.
    """
    
    def __init__(self):
        self.preferences: List[PreferenceExample] = []
    
    def add_preference(self, pair: ContentPair):
        """Add a preference from a content pair."""
        if pair.winner is None:
            logger.warning("Pair has no winner, skipping")
            return
        
        if pair.winner == 'a':
            chosen = pair.variation_a
            rejected = pair.variation_b
        else:
            chosen = pair.variation_b
            rejected = pair.variation_a
        
        self.preferences.append(PreferenceExample(
            prompt=pair.prompt,
            chosen=chosen,
            rejected=rejected,
            reason=f"Higher {pair.metrics.get('optimization_goal', 'performance')}"
        ))
    
    def create_training_data(self) -> List[Dict[str, str]]:
        """
        Format training data for DPO fine-tuning.
        
        Format required by DPO:
        {
            "prompt": "...",
            "chosen": "...",
            "rejected": "..."
        }
        """
        training_data = []
        
        for pref in self.preferences:
            training_data.append({
                "prompt": pref.prompt,
                "chosen": pref.chosen,
                "rejected": pref.rejected
            })
        
        return training_data
    
    def save_training_data(self, filepath: str):
        """Save training data to JSONL file."""
        training_data = self.create_training_data()
        
        with open(filepath, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved {len(training_data)} preference examples to {filepath}")


# ============================================================================
# STEP 4: Perform Preference Tuning
# ============================================================================

class PreferenceTuner:
    """
    STEP 4: Perform Preference Tuning
    
    Fine-tunes model using DPO (Direct Preference Optimization).
    """
    
    def __init__(self, base_model: str = "gpt-3.5-turbo"):
        self.base_model = base_model
        self.fine_tuned_model_id = None
    
    def fine_tune(self, training_file: str):
        """
        Fine-tune model using DPO.
        
        In production, this would:
        1. Load base model and tokenizer
        2. Load preference dataset
        3. Configure DPO trainer
        4. Train model
        5. Save fine-tuned model
        """
        logger.info(f"Preparing DPO fine-tuning for {self.base_model}")
        logger.info(f"Training file: {training_file}")
        
        # In production using transformers and TRL:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # from trl import DPOConfig, DPOTrainer
        # from datasets import load_dataset
        #
        # model = AutoModelForCausalLM.from_pretrained(base_model)
        # tokenizer = AutoTokenizer.from_pretrained(base_model)
        # dataset = load_dataset('json', data_files=training_file, split='train')
        #
        # dpo_config = DPOConfig(
        #     output_dir="content-optimized-model",
        #     per_device_train_batch_size=4,
        #     learning_rate=1e-5
        # )
        #
        # trainer = DPOTrainer(
        #     model=model,
        #     args=dpo_config,
        #     train_dataset=dataset,
        #     tokenizer=tokenizer
        # )
        #
        # trainer.train()
        # trainer.save_model("content-optimized-model")
        
        logger.info("DPO fine-tuning would proceed here...")
        logger.info("After completion, fine-tuned model would be saved")
        
        # Simulate fine-tuned model ID
        self.fine_tuned_model_id = f"ft-{self.base_model}-optimized"
    
    def validate(self, validation_pairs: List[ContentPair]) -> Dict[str, float]:
        """Validate fine-tuned model on held-out examples."""
        logger.info(f"Validating fine-tuned model on {len(validation_pairs)} examples")
        
        # In production, would test model and compute metrics
        return {
            "win_rate": 0.75,  # Model generates winning content 75% of the time
            "improvement": 0.25,  # 25% improvement over base model
            "consistency": 0.88
        }


# ============================================================================
# STEP 5: Use Tuned LLM
# ============================================================================

class OptimizedContentGenerator:
    """
    STEP 5: Use Tuned LLM
    
    Uses fine-tuned model to generate optimized content.
    """
    
    def __init__(self, fine_tuned_model_id: str):
        self.fine_tuned_model_id = fine_tuned_model_id
    
    def generate_optimized(self, prompt: str) -> str:
        """
        Generate optimized content using fine-tuned model.
        
        In production, this would call the fine-tuned model.
        """
        # In production:
        # response = fine_tuned_model.generate(
        #     prompt=prompt,
        #     max_tokens=100
        # )
        # return response
        
        logger.info(f"Using optimized model {self.fine_tuned_model_id}")
        # Simulate optimized output (would be better than base model)
        return f"[Optimized content for: {prompt}]"


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_step_1():
    """Demonstrate Step 1: Generate Pair of Contents."""
    print("\n" + "="*70)
    print("STEP 1: Generate Pair of Contents")
    print("="*70)
    
    print("\n💡 Why Generate Pairs?")
    print("   • Get diverse variations from same prompt")
    print("   • Use different sampling strategies")
    print("   • Enable head-to-head comparison")
    
    generator = ContentGenerator()
    
    print("\n📝 Example: Email Subject Lines")
    prompt = "New product launch announcement"
    
    var_a, var_b = generator.generate_pair(prompt, "subject_line")
    
    print(f"\n   Prompt: {prompt}")
    print(f"   Variation A: {var_a}")
    print(f"   Variation B: {var_b}")
    
    print("\n✅ Two variations generated - ready for comparison")


def demonstrate_step_2():
    """Demonstrate Step 2: Compare and Pick Winner."""
    print("\n" + "="*70)
    print("STEP 2: Compare and Pick Winner")
    print("="*70)
    
    print("\n📊 Comparison Process:")
    print("   • Test both variations")
    print("   • Measure optimization metric (e.g., open rate)")
    print("   • Select winner based on performance")
    
    generator = ContentGenerator()
    comparator = ContentComparator(optimization_goal="open_rate")
    
    prompt = "Weekly newsletter"
    pair = ContentPair(
        prompt=prompt,
        variation_a="Your Weekly Update is Here",
        variation_b="What's New This Week? (You'll Want to See This)"
    )
    
    pair = comparator.compare(pair)
    
    print(f"\n   Variation A: {pair.variation_a}")
    print(f"   Metrics: {pair.metrics['variation_a']:.2%} open rate")
    print(f"\n   Variation B: {pair.variation_b}")
    print(f"   Metrics: {pair.metrics['variation_b']:.2%} open rate")
    print(f"\n   🏆 Winner: Variation {pair.winner.upper()}")
    
    print("\n✅ Winner selected - ready for training dataset")


def demonstrate_step_3():
    """Demonstrate Step 3: Create Training Dataset."""
    print("\n" + "="*70)
    print("STEP 3: Create Training Dataset")
    print("="*70)
    
    print("\n📚 Collecting Preference Pairs")
    print("   Process:")
    print("   1. Generate pairs for multiple prompts")
    print("   2. Compare and identify winners")
    print("   3. Format as (prompt, chosen, rejected)")
    
    # Simulate collecting preferences
    builder = PreferenceDatasetBuilder()
    
    examples = [
        ContentPair(
            prompt="Product launch",
            variation_a="New Product: Check It Out!",
            variation_b="You're Invited: Exclusive Preview",
            winner="b",
            metrics={"variation_a": 0.45, "variation_b": 0.62}
        ),
        ContentPair(
            prompt="Newsletter",
            variation_a="Your Weekly Update",
            variation_b="What's New This Week?",
            winner="b",
            metrics={"variation_a": 0.38, "variation_b": 0.55}
        ),
    ]
    
    for pair in examples:
        builder.add_preference(pair)
    
    training_data = builder.create_training_data()
    
    print(f"\n   Collected {len(training_data)} preference pairs")
    print("\n   Training data format:")
    print(json.dumps(training_data[0], indent=2))
    
    print("\n✅ Training dataset created - ready for fine-tuning")
    print("   (In production, collect 100+ high-quality pairs)")


def demonstrate_step_4():
    """Demonstrate Step 4: Perform Preference Tuning."""
    print("\n" + "="*70)
    print("STEP 4: Perform Preference Tuning (DPO)")
    print("="*70)
    
    print("\n🔧 DPO Fine-Tuning Process:")
    print("   1. Load base model and tokenizer")
    print("   2. Load preference dataset")
    print("   3. Configure DPO trainer")
    print("   4. Train model on preferences")
    print("   5. Save fine-tuned model")
    
    tuner = PreferenceTuner()
    
    print("\n   Preparing DPO fine-tuning...")
    tuner.fine_tune("preference_dataset.jsonl")
    
    print(f"\n   Fine-tuned model ID: {tuner.fine_tuned_model_id}")
    
    # Validate
    validation_pairs = [
        ContentPair("Test prompt 1", "Var A", "Var B", "a"),
        ContentPair("Test prompt 2", "Var A", "Var B", "b"),
    ]
    
    metrics = tuner.validate(validation_pairs)
    print("\n   Validation Metrics:")
    print(f"   • Win Rate: {metrics['win_rate']:.2%}")
    print(f"   • Improvement: {metrics['improvement']:.2%}")
    print(f"   • Consistency: {metrics['consistency']:.2%}")
    
    print("\n✅ Model fine-tuned - ready for optimized generation")


def demonstrate_step_5():
    """Demonstrate Step 5: Use Tuned LLM."""
    print("\n" + "="*70)
    print("STEP 5: Use Tuned LLM")
    print("="*70)
    
    print("\n🚀 Generating Optimized Content")
    
    generator = OptimizedContentGenerator("ft-gpt-3.5-turbo-optimized")
    
    prompts = [
        "New feature announcement",
        "Weekly newsletter",
        "Product update"
    ]
    
    print("\n   Using optimized model to generate content:")
    for prompt in prompts:
        result = generator.generate_optimized(prompt)
        print(f"   • {prompt}: {result}")
    
    print("\n✅ Optimized content generated - better performance expected")


def show_comparison():
    """Show comparison with traditional A/B testing."""
    print("\n" + "="*70)
    print("⚖️  COMPARISON: Traditional A/B Testing vs Content Optimization")
    print("="*70)
    
    print("\n📊 Traditional A/B Testing:")
    print("   ✅ Simple and straightforward")
    print("   ✅ Works with clear hypotheses")
    print("   ⚠️  Manual variant creation")
    print("   ⚠️  Limited number of tests")
    print("   ⚠️  Doesn't learn patterns")
    print("   ⚠️  Time-consuming")
    
    print("\n🤖 Content Optimization (DPO):")
    print("   ✅ Automated content generation")
    print("   ✅ Learns from all comparisons")
    print("   ✅ Scales to many variations")
    print("   ✅ Model internalizes patterns")
    print("   ✅ Continuous improvement")
    print("   ⚠️  Requires training data")
    print("   ⚠️  More complex setup")
    
    print("\n💡 Key Difference:")
    print("   A/B Testing: Test specific hypotheses")
    print("   Content Optimization: Learn what works and generate it")


def show_real_world_example():
    """Show realistic use case example."""
    print("\n" + "="*70)
    print("🌍 REAL-WORLD USE CASE: Email Subject Line Optimization")
    print("="*70)
    
    print("\nScenario: Marketing team needs to optimize email subject lines")
    print("for maximum open rates")
    
    print("\n📝 Challenge:")
    print("   • Manual A/B testing is slow")
    print("   • Limited number of variations to test")
    print("   • Don't know what patterns work best")
    print("   • Need consistent high-performing content")
    
    print("\n✅ Solution: Content Optimization")
    
    print("\n   Step 1: Generate pairs")
    print("   • 'New Product Launch: Check It Out!'")
    print("   • 'You're Invited: Exclusive Preview Inside'")
    
    print("\n   Step 2: Compare (test open rates)")
    print("   • Variation A: 45% open rate")
    print("   • Variation B: 62% open rate")
    print("   • Winner: Variation B")
    
    print("\n   Step 3-4: Collect 100+ pairs, fine-tune with DPO")
    print("   • Model learns: curiosity-driven, personal tone wins")
    print("   • Model learns: questions and exclusivity work well")
    
    print("\n   Step 5: Use optimized model")
    print("   • Generates subject lines that consistently win")
    print("   • Average open rate improves from 45% to 58%")
    
    print("\n" + "="*70)
    print("🎯 Impact: Higher open rates, faster content creation, data-driven decisions")
    print("="*70)


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 CONTENT OPTIMIZATION PATTERN")
    print("="*70)
    
    print("\n📋 Pattern Overview:")
    print("   Optimize content for specific goals using preference-based fine-tuning")
    print("   Learn from pairwise comparisons to generate better content")
    print("   Use DPO to train model on what wins")
    
    # Demonstrate all steps
    demonstrate_step_1()
    demonstrate_step_2()
    demonstrate_step_3()
    demonstrate_step_4()
    demonstrate_step_5()
    
    # Show comparisons and use cases
    show_comparison()
    show_real_world_example()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for detailed explanation")
    print("   2. Define your optimization goal")
    print("   3. Generate and compare content pairs")
    print("   4. Build preference dataset")
    print("   5. Fine-tune with DPO")
    print("   6. Use optimized model for better content")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

