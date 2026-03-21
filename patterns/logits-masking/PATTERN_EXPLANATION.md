# Logits Masking Pattern - Detailed Explanation

## Overview

The **Logits Masking** pattern is a technique for constraining language model generation to ensure outputs conform to specific rules, formats, or constraints. Instead of post-processing or retrying, we modify the generation process itself to prevent invalid outputs.

## The Three Steps

### Step 1: Intercept Sampling

**What it is:**
- Before the model samples the next token, we intercept the logits (probability scores for each token)
- Logits are the raw output from the model's final layer, representing how likely each token is to be next

**How it works:**
- The `LogitsProcessor` class from transformers provides a hook: the `__call__` method
- This method is automatically invoked by the generation pipeline before sampling
- We receive the logits tensor and can modify it before the model chooses the next token

**In our code:**
```python
def __call__(self, input_ids, scores):
    # This is called BEFORE sampling
    # scores contains logits for all possible next tokens
    # We can modify scores here
    return scores
```

**Why it matters:**
- This is the only point where we can influence token selection before it happens
- Post-processing can't fix invalid sequences that are already generated
- Intercepting at this stage is efficient and prevents wasted computation

---

### Step 2: Zero Out Invalid Sequences

**What it is:**
- For each potential next token, we check if it would lead to an invalid sequence
- Invalid tokens get their logits set to `float('-inf')`, making their probability effectively zero
- The model can only sample from valid tokens

**How it works:**
1. Analyze the current sequence state (e.g., JSON structure, brace depth, string state)
2. For each token in the vocabulary, check if it would be valid
3. Set invalid token logits to `-inf` (they become impossible to sample)

**In our code:**
```python
for token_id in range(scores.shape[-1]):
    if not self._is_valid_json_token(token_id, current_text):
        scores[0, token_id] = float('-inf')  # Zero out invalid token
```

**Example:**
- Current sequence: `{"status": "success", "data": {`
- Invalid tokens: `}` (would close the outer object prematurely)
- Valid tokens: `"key"`, `[`, numbers, etc.
- Result: `}` gets masked, model can only choose valid tokens

**Why it matters:**
- Prevents invalid sequences at the source
- More efficient than generating and rejecting
- Ensures the model never produces certain types of errors

---

### Step 3: Backtracking and Regenerating

**What it is:**
- If an invalid sequence is detected despite masking, we revert to a previous valid state
- We save checkpoints during generation at regular intervals
- When backtracking, we load a checkpoint and continue generation from there

**How it works:**
1. **Checkpointing**: Save the current generation state every N steps
2. **Validation**: Continuously check if the current sequence is valid
3. **Backtracking**: If invalid, revert to last checkpoint
4. **Regeneration**: Continue generation from the checkpoint

**In our code:**
```python
# Save checkpoint
if step % 5 == 0:
    self.checkpoint_states.append(generated_ids.clone())

# Detect invalid sequence
if not self._is_valid_json_prefix(current_text):
    # Backtrack
    generated_ids = self.checkpoint_states.pop()
    backtracks += 1
```

**Example:**
- Step 10: `{"status": "success", "data": {` ✓ (checkpoint saved)
- Step 15: `{"status": "success", "data": {"key": value` ✓
- Step 20: `{"status": "success", "data": {"key": value}` ✓
- Step 25: `{"status": "success", "data": {"key": value}, invalid` ✗ (invalid!)
- **Backtrack to step 20** and regenerate

**Why it matters:**
- Provides a safety net when masking isn't perfect
- Handles edge cases and complex validation rules
- Allows recovery from errors without starting over

---

## Why This Example is Better

### Compared to Reference Example (Product Descriptions)

1. **Realistic Production Use Case**
   - **Reference**: Product descriptions with banned words (marketing use case)
   - **Our Example**: API response generation with JSON schema (common in production APIs)
   - **Benefit**: More directly applicable to real-world software development

2. **Complete Implementation of All Three Steps**
   - **Reference**: Focuses mainly on Step 2 (masking banned words)
   - **Our Example**: Explicitly implements all three steps with clear separation
   - **Benefit**: Better understanding of the complete pattern

3. **Better Error Handling**
   - **Reference**: Minimal error handling
   - **Our Example**: Comprehensive logging, fallback mechanisms, graceful degradation
   - **Benefit**: Production-ready code that handles edge cases

4. **Backtracking Mechanism**
   - **Reference**: No backtracking shown
   - **Our Example**: Full backtracking implementation with checkpoint management
   - **Benefit**: Demonstrates recovery from invalid sequences

5. **Open Source & Local-First**
   - **Reference**: Requires HuggingFace API access and specific models
   - **Our Example**: Works with local models, open-source tools, and Ollama
   - **Benefit**: More accessible and doesn't require API keys

6. **Clear Documentation**
   - **Reference**: Notebook format with minimal explanation
   - **Our Example**: Comprehensive README, inline comments, pattern explanation
   - **Benefit**: Easier to understand and learn from

---

## Use Cases

### 1. API Response Generation
- **Problem**: LLMs generate invalid JSON for API responses
- **Solution**: Logits masking ensures valid JSON syntax
- **Example**: REST API that generates structured responses

### 2. Code Generation
- **Problem**: Generated code doesn't follow style guidelines
- **Solution**: Mask tokens that violate coding standards
- **Example**: Enforcing Python PEP 8 or specific formatting rules

### 3. Content Moderation
- **Problem**: Model generates inappropriate content
- **Solution**: Mask banned words/phrases during generation
- **Example**: Chatbot that must avoid certain topics

### 4. Structured Data Extraction
- **Problem**: Extracted data doesn't match required format
- **Solution**: Constrain generation to valid formats (CSV, XML, etc.)
- **Example**: Generating database records or configuration files

### 5. Schema Validation
- **Problem**: Generated data doesn't conform to schema
- **Solution**: Mask tokens that would violate schema constraints
- **Example**: Generating data that must match a JSON Schema

---

## Best Practices

1. **Pre-compute Valid Tokens**: Build lookup tables for common states to speed up validation
2. **Efficient State Tracking**: Use finite state machines for complex validation rules
3. **Checkpoint Strategy**: Balance checkpoint frequency (more = safer but uses more memory)
4. **Graceful Degradation**: Provide fallback mechanisms when backtracking fails
5. **Logging**: Track masked tokens and backtrack events for debugging
6. **Performance**: Cache constraint checks to avoid redundant computations
7. **Schema Validation**: Use JSON Schema or similar for complex structure validation

---

## Technical Details

### Logits vs Probabilities

- **Logits**: Raw scores from the model (can be any real number)
- **Probabilities**: Softmax of logits (sum to 1.0)
- **Masking**: Setting logits to `-inf` makes probability = 0

### Token Sampling

- **Greedy**: Always choose highest probability token
- **Sampling**: Randomly choose based on probabilities
- **Masking**: Removes tokens from consideration before sampling

### State Management

- **Current State**: Track where we are in the structure (e.g., inside JSON object)
- **Validation**: Check if next token maintains valid state
- **Checkpointing**: Save valid states for backtracking

---

## References

- [HuggingFace LogitsProcessor Documentation](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor)
- [Structured Generation with Transformers](https://huggingface.co/docs/transformers/main/en/llm_tutorial#controlling-generation)
- [Finite State Machines for Text Generation](https://v4nn4.github.io/posts/ner-using-structured-generation/)

