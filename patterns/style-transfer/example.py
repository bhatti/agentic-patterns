"""
Style Transfer Pattern

This example demonstrates two approaches to style transfer:

OPTION 1: Few-Shot Learning (In-Context Learning)
    - Provide example pairs in the prompt
    - Model learns pattern from examples
    - No training required

OPTION 2: Model Fine-Tuning
    - Fine-tune model on style pairs dataset
    - Model learns style transformation
    - Better consistency, requires training data

USE CASE: Notes to Professional Email
    Realistic scenario: Converting informal meeting notes or quick
    jottings into professional, well-structured emails.
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
class StyleExample:
    """Represents a style transfer example pair."""
    input_text: str
    output_text: str
    description: Optional[str] = None


# ============================================================================
# OPTION 1: Few-Shot Learning (In-Context Learning)
# ============================================================================

class FewShotStyleTransfer:
    """
    OPTION 1: Few-Shot Learning
    
    Uses example pairs in the prompt to teach the model the style transformation.
    """
    
    def __init__(self, examples: List[StyleExample], style_description: str = ""):
        """
        Initialize with style examples.
        
        Args:
            examples: List of example pairs showing style transformation
            style_description: Optional description of the target style
        """
        self.examples = examples
        self.style_description = style_description
    
    def create_prompt(self, input_text: str) -> str:
        """
        Create a few-shot prompt with examples.
        
        Step 1: Format examples into prompt
        Step 2: Add style description if provided
        Step 3: Include target text
        """
        prompt_parts = []
        
        # Add style description
        if self.style_description:
            prompt_parts.append(f"Style: {self.style_description}\n")
        
        prompt_parts.append("Convert the following informal notes into professional emails.\n")
        prompt_parts.append("Examples:\n")
        
        # Add example pairs
        for i, example in enumerate(self.examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"  Notes: {example.input_text}")
            prompt_parts.append(f"  Email: {example.output_text}")
            prompt_parts.append("")
        
        # Add target text
        prompt_parts.append("Now convert this:")
        prompt_parts.append(f"  Notes: {input_text}")
        prompt_parts.append("  Email:")
        
        return "\n".join(prompt_parts)
    
    def transfer_style(self, input_text: str, model=None) -> str:
        """
        Transfer style using few-shot learning.
        
        In production, this would call an actual LLM.
        For demonstration, we simulate the process.
        """
        prompt = self.create_prompt(input_text)
        
        # In production, this would be:
        # response = model.generate(prompt)
        # return response
        
        # For demonstration, show what would be generated
        logger.info("Few-shot prompt created")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Simulate style transfer based on examples
        return self._simulate_transfer(input_text)
    
    def _simulate_transfer(self, input_text: str) -> str:
        """Simulate style transfer for demonstration."""
        # Extract key information from input
        # In real implementation, model would do this
        
        # Simple simulation: format as professional email
        lines = input_text.split('\n')
        subject = ""
        body_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect urgency
            if 'urgent' in line.lower() or 'asap' in line.lower():
                subject = "Urgent: " + line.split(':')[-1].strip() if ':' in line else line
            elif 'subject' in line.lower() or 're:' in line.lower():
                subject = line.replace('subject:', '').replace('re:', '').strip()
            else:
                body_parts.append(line)
        
        # Format as professional email
        email = f"""Subject: {subject or 'Follow-up'}

Dear [Recipient],

"""
        
        for part in body_parts:
            # Clean up informal language
            part = part.replace('ur', 'your').replace('u ', 'you ').replace('2', 'to')
            part = part.replace('rdy', 'ready').replace('asap', 'as soon as possible')
            email += f"{part.capitalize()}.\n\n"
        
        email += """Best regards,
[Your Name]"""
        
        return email


# ============================================================================
# OPTION 2: Model Fine-Tuning
# ============================================================================

class FineTunedStyleTransfer:
    """
    OPTION 2: Model Fine-Tuning
    
    Uses a fine-tuned model trained on style pairs.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model (optional for demo)
        """
        self.model_path = model_path
        self.is_fine_tuned = model_path is not None
    
    def prepare_training_data(self, examples: List[StyleExample]) -> List[Dict]:
        """
        Prepare training data for fine-tuning.
        
        Format: List of prompt-completion pairs
        """
        training_data = []
        
        for example in examples:
            training_data.append({
                "prompt": f"Convert notes to professional email:\nNotes: {example.input_text}\nEmail:",
                "completion": example.output_text
            })
        
        return training_data
    
    def fine_tune_model(self, training_data: List[Dict], base_model: str = "gpt-3.5-turbo"):
        """
        Fine-tune model on style pairs.
        
        In production, this would:
        1. Format data for fine-tuning API
        2. Upload training file
        3. Create fine-tuning job
        4. Monitor training
        5. Use fine-tuned model
        """
        logger.info(f"Preparing to fine-tune {base_model} on {len(training_data)} examples")
        logger.info("Training data format:")
        logger.info(json.dumps(training_data[0], indent=2))
        
        # In production:
        # 1. Save training data to JSONL
        # 2. Upload to fine-tuning service
        # 3. Create fine-tuning job
        # 4. Wait for completion
        # 5. Use fine-tuned model ID
        
        logger.info("Fine-tuning would proceed here...")
        logger.info("After fine-tuning, use the fine-tuned model for inference")
    
    def transfer_style(self, input_text: str) -> str:
        """
        Transfer style using fine-tuned model.
        
        In production, this would call the fine-tuned model.
        """
        if not self.is_fine_tuned:
            logger.warning("Model not fine-tuned. Use fine_tune_model() first.")
            return ""
        
        # In production:
        # response = fine_tuned_model.generate(
        #     prompt=f"Convert notes to professional email:\nNotes: {input_text}\nEmail:"
        # )
        # return response
        
        # For demonstration
        logger.info("Using fine-tuned model for style transfer")
        return f"[Fine-tuned model output for: {input_text}]"


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_option_1():
    """Demonstrate Option 1: Few-Shot Learning."""
    print("\n" + "="*70)
    print("OPTION 1: Few-Shot Learning (In-Context Learning)")
    print("="*70)
    
    print("\n📚 Step 1: Collect Style Examples")
    examples = [
        StyleExample(
            input_text="urgent: need meeting minutes by friday\nfor stakeholder presentation\ndetails:\n- This is needed for our upcoming presi to stakeholders.\n- will need ur help asap\n- send 2 me when rdy",
            output_text="""Subject: Urgent: Meeting Minutes Needed for Stakeholder Presentation

Dear [Recipient],

I hope this message finds you well. This is a gentle reminder that we require the meeting minutes for the stakeholder presentation scheduled for this week by Friday. Your prompt assistance in preparing the minutes is highly appreciated.

Please ensure that the meeting minutes are comprehensive and accurately reflect the discussions held during the meeting. Once the minutes are ready, kindly send them to me for review at your earliest convenience.

Thank you for your attention to this matter. Should you have any questions or require further clarification, please do not hesitate to reach out.

Best regards,
[Your Name]""",
            description="Urgent request with informal language"
        ),
        StyleExample(
            input_text="meeting tomorrow 2pm\nagenda: q4 planning\nneed: budget numbers, team updates\nlocation: conf room b",
            output_text="""Subject: Meeting Reminder: Q4 Planning Session

Dear Team,

I would like to remind you of our upcoming meeting scheduled for tomorrow at 2:00 PM in Conference Room B.

Agenda:
- Q4 Planning Discussion
- Budget Review
- Team Updates

Please come prepared with the budget numbers and your team updates. Your participation is important for our planning process.

Looking forward to our discussion.

Best regards,
[Your Name]""",
            description="Meeting reminder with abbreviations"
        ),
        StyleExample(
            input_text="thx for the help on the project!\nreally appreciate it\nlets discuss next steps soon",
            output_text="""Subject: Thank You and Next Steps Discussion

Dear [Recipient],

I wanted to take a moment to express my sincere gratitude for your valuable assistance on the project. Your contributions have been instrumental in our progress, and I truly appreciate your dedication and expertise.

I would like to schedule a time to discuss our next steps and how we can continue to move forward effectively. Please let me know your availability, and I will coordinate a meeting that works for everyone.

Thank you again for your support.

Best regards,
[Your Name]""",
            description="Thank you note with informal language"
        )
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n   Example {i}:")
        print(f"   Input: {ex.input_text[:60]}...")
        print(f"   Output: {ex.output_text[:60]}...")
    
    print("\n🔧 Step 2: Create Few-Shot Prompt")
    transfer = FewShotStyleTransfer(examples)
    
    test_input = """quick update: project deadline moved to next week
need to adjust timeline
can we meet today to discuss?"""
    
    prompt = transfer.create_prompt(test_input)
    print(f"\n   Prompt created ({len(prompt)} characters)")
    print("   Structure:")
    print("   - Style description")
    print("   - Example pairs (3 examples)")
    print("   - Target text to convert")
    
    print("\n⚙️  Step 3: Generate Style-Transferred Output")
    output = transfer.transfer_style(test_input)
    print("\n   Input (Notes):")
    print(f"   {test_input}")
    print("\n   Output (Professional Email):")
    print(f"   {output}")
    
    print("\n✅ Advantages:")
    print("   • No training required")
    print("   • Quick to implement")
    print("   • Easy to adjust by changing examples")
    print("   • Works with any LLM")


def demonstrate_option_2():
    """Demonstrate Option 2: Model Fine-Tuning."""
    print("\n" + "="*70)
    print("OPTION 2: Model Fine-Tuning")
    print("="*70)
    
    print("\n📊 Step 1: Prepare Training Dataset")
    examples = [
        StyleExample(
            input_text="urgent: need meeting minutes by friday",
            output_text="Subject: Urgent: Meeting Minutes Needed\n\nDear [Recipient],\n\nI hope this message finds you well. This is a gentle reminder that we require the meeting minutes by Friday. Your prompt assistance is highly appreciated.\n\nBest regards,\n[Your Name]"
        ),
        StyleExample(
            input_text="meeting tomorrow 2pm\nagenda: q4 planning",
            output_text="Subject: Meeting Reminder: Q4 Planning Session\n\nDear Team,\n\nI would like to remind you of our upcoming meeting scheduled for tomorrow at 2:00 PM.\n\nAgenda: Q4 Planning Discussion\n\nBest regards,\n[Your Name]"
        ),
        # In production, you'd have 100+ examples
    ]
    
    print(f"   Collected {len(examples)} example pairs")
    print("   (In production, collect 100+ high-quality pairs)")
    
    print("\n🔧 Step 2: Format Training Data")
    transfer = FineTunedStyleTransfer()
    training_data = transfer.prepare_training_data(examples)
    
    print("   Training data format:")
    print(json.dumps(training_data[0], indent=2))
    
    print("\n⚙️  Step 3: Fine-Tune Model")
    print("   Process:")
    print("   1. Save training data to JSONL file")
    print("   2. Upload to fine-tuning service (OpenAI, HuggingFace, etc.)")
    print("   3. Create fine-tuning job")
    print("   4. Monitor training progress")
    print("   5. Use fine-tuned model for inference")
    
    transfer.fine_tune_model(training_data)
    
    print("\n✅ Advantages:")
    print("   • Better style consistency")
    print("   • Handles complex transformations")
    print("   • More efficient at inference")
    print("   • Learns subtle style nuances")
    
    print("\n⚠️  Considerations:")
    print("   • Requires training data (100+ examples recommended)")
    print("   • More complex setup")
    print("   • Less flexible (need retraining to change style)")


def compare_approaches():
    """Compare the two approaches."""
    print("\n" + "="*70)
    print("⚖️  COMPARISON: Few-Shot vs Fine-Tuning")
    print("="*70)
    
    print("\n1️⃣  Few-Shot Learning")
    print("   ✅ No training required")
    print("   ✅ Quick to implement (minutes)")
    print("   ✅ Easy to adjust (change examples)")
    print("   ✅ Works with any LLM")
    print("   ⚠️  Limited by context window")
    print("   ⚠️  May not capture complex nuances")
    
    print("\n2️⃣  Fine-Tuning")
    print("   ✅ Better consistency")
    print("   ✅ Handles complex transformations")
    print("   ✅ More efficient inference")
    print("   ✅ Learns subtle nuances")
    print("   ⚠️  Requires training data")
    print("   ⚠️  More complex setup")
    print("   ⚠️  Less flexible")
    
    print("\n💡 When to Use:")
    print("   • Few-Shot: Quick prototypes, changing styles, limited data")
    print("   • Fine-Tuning: Production systems, consistent style, large datasets")


def show_real_world_example():
    """Show realistic use case example."""
    print("\n" + "="*70)
    print("🌍 REAL-WORLD USE CASE: Notes to Professional Email")
    print("="*70)
    
    print("\nScenario: Converting informal meeting notes into professional emails")
    print("Challenge: Maintain meaning while transforming style and tone")
    
    print("\n📝 Input (Informal Notes):")
    informal = """quick update: project deadline moved to next week
need to adjust timeline
can we meet today to discuss?
thx!"""
    print(f"   {informal}")
    
    print("\n❌ Without Style Transfer:")
    print("   • Manual rewriting required")
    print("   • Inconsistent tone")
    print("   • Time-consuming")
    print("   • May miss important details")
    
    print("\n✅ With Style Transfer (Few-Shot):")
    transfer = FewShotStyleTransfer([
        StyleExample(
            input_text="meeting tomorrow 2pm",
            output_text="Subject: Meeting Reminder\n\nDear Team,\n\nI would like to remind you of our meeting scheduled for tomorrow at 2:00 PM.\n\nBest regards,\n[Your Name]"
        )
    ])
    
    output = transfer.transfer_style(informal)
    print(f"   {output}")
    
    print("\n   Benefits:")
    print("   • Automated transformation")
    print("   • Consistent professional tone")
    print("   • Preserves all information")
    print("   • Ready to send")
    
    print("\n" + "="*70)
    print("🎯 Impact: Save time, ensure consistency, maintain professionalism")
    print("="*70)


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 STYLE TRANSFER PATTERN")
    print("="*70)
    
    print("\n📋 Pattern Overview:")
    print("   Transform content from one style to another")
    print("   Two approaches: Few-shot learning and Fine-tuning")
    print("   Preserve meaning while changing tone and format")
    
    # Demonstrate both options
    demonstrate_option_1()
    demonstrate_option_2()
    
    # Show comparisons and use cases
    compare_approaches()
    show_real_world_example()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for detailed explanation")
    print("   2. Choose approach based on your needs")
    print("   3. Collect quality examples for your use case")
    print("   4. For production, use actual LLM APIs or local models")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

