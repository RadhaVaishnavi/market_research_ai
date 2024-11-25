from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def generate_use_cases(industry_name, insights):
    """
    Generate AI/GenAI use cases using a Hugging Face GPT-2 model and enforce proper formatting.
    """
    # Create a structured prompt
    prompt = (
        f"Generate AI/GenAI use cases for the {industry_name} industry based on the following insights:\n"
        f"{insights}\n\n"
        "Each use case should follow this format:\n"
        "Use Case: [Title]\n"
        "AI Application: [Detailed description]\n"
        "Cross-Functional Benefit: [List benefits across teams/functions]\n\n"
        "Ensure there are no extra spaces or formatting issues."
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )

    # Decode the model output
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to ensure clean output
    formatted_text = raw_text.replace("\n\n", "\n").strip()  # Clean up excessive line breaks
    formatted_text = ' '.join(formatted_text.split())  # Remove excessive spaces
    return formatted_text

# Example usage with clear insights
insights = "Five trends converging for AI-enabled healthcare: Why AI and..."
use_cases = generate_use_cases("healthcare", insights)
print(use_cases)
