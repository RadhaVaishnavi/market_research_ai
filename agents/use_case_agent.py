from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def generate_use_cases(industry_name, insights):
    """
    Generate AI/GenAI use cases using a Hugging Face GPT-2 model and return structured output.
    """
    # Create a prompt for structured output
    prompt = (
        f"Generate detailed and structured AI/GenAI use cases for the {industry_name} industry based on the following insights:\n"
        f"{insights}\n\n"
        "Each use case should include:\n"
        "1. Use Case\n"
        "2. AI Application\n"
        "3. Cross-Functional Benefit\n\n"
        "Format the output as:\n"
        "Use Case: [Brief title of the use case]\n"
        "AI Application: [Explanation of AI application in this use case]\n"
        "Cross-Functional Benefit: [Benefits across various teams or functions]\n\n"
        "Provide at least one use case."
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=300,  # Set appropriate length for detailed output
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )

    # Decode and clean up the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Ensure proper formatting by removing unintended line breaks
    cleaned_text = "\n".join([line.strip() for line in generated_text.splitlines() if line.strip()])

    return cleaned_text
