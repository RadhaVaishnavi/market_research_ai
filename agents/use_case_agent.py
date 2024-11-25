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
        "Each use case should follow this format without any extra spaces:\n"
        "Use Case: [Title]\n"
        "AI Application: [Detailed description]\n"
        "Cross-Functional Benefit: [List benefits across teams/functions]\n\n"
        "Example:\n"
        "Use Case: AI-Powered Predictive Maintenance\n"
        "AI Application: Implement machine learning algorithms that analyze real-time sensor data from hospital equipment to predict potential failures and schedule maintenance proactively.\n"
        "Cross-Functional Benefit: Operations: Minimizes unplanned downtime. Finance: Reduces maintenance costs.\n\n"
        "Please provide clear responses without any extra spacing or formatting issues."
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
    formatted_text = ' '.join(raw_text.split())  # Clean up excessive spaces
    return formatted_text.strip()

# Example usage
use_cases = generate_use_cases("Healthcare", "AI can improve patient diagnosis and streamline administrative tasks.")
print(use_cases)
