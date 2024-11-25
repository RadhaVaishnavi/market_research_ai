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
        "Ensure the output is clean, detailed, and avoids bullet points or line-by-line formatting issues."
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )

    # Decode and clean up the response
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to ensure correct formatting
    formatted_text = ""
    sections = raw_text.split("Use Case:")  # Split by expected section header
    for section in sections:
        if section.strip():  # Process non-empty sections
            lines = section.strip().split("\n")
            title = f"Use Case: {lines[0].strip()}" if lines else "Use Case: N/A"
            ai_app = "AI Application: " + next((line.strip() for line in lines if "AI Application" in line), "N/A")
            benefits = "Cross-Functional Benefit: " + next((line.strip() for line in lines if "Cross-Functional Benefit" in line), "N/A")
            formatted_text += f"{title}\n{ai_app}\n{benefits}\n\n"

    return formatted_text.strip()
