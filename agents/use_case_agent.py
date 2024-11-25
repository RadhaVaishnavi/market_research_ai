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
        f"Generate structured AI/GenAI use cases for the {industry_name} industry based on the following insights:\n"
        f"{insights}\n\n"
        "Each use case should follow this format:\n"
        "Use Case: [Title]\n"
        "AI Application: [Detailed description]\n"
        "Cross-Functional Benefit: [List benefits across teams/functions]\n\n"
        "Please provide at least three distinct use cases."
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=500,  # Increased length to accommodate multiple use cases
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
    formatted_text = raw_text.replace("\n", "\n").strip()  # Retain line breaks for formatting
    return formatted_text

# Example usage
use_cases = generate_use_cases("Healthcare", "AI can improve patient diagnosis and streamline administrative tasks.")
print(use_cases)
