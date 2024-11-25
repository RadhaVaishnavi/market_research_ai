from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def generate_use_cases(industry_name, insights):
    """
    Generate AI/GenAI use cases using a Hugging Face GPT-2 model.
    """
    # Create a prompt based on the inputs
    prompt = f"Generate AI/GenAI use cases for the {industry_name} industry based on the following insights: {insights}"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoid repetition of phrases
        do_sample=True,  # Use sampling to get diverse results
        top_k=50,  # Consider the top 50 words at each step
        top_p=0.9,  # Use nucleus sampling
        temperature=0.7  # Control randomness
    )

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up and return the result
    return generated_text.strip()
