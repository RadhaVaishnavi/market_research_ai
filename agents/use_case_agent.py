# Import required modules from transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer from Hugging Face (no need for OpenAI API key)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def generate_use_cases(industry_name, insights):
    """
    Generate AI/GenAI use cases using a language model (GPT-2 from Hugging Face).
    industry_name: The name of the industry to tailor use cases for.
    insights: Key insights or requirements about the industry or company.
    """
    # Formulate the prompt based on the input industry and insights
    prompt = f"Generate AI/GenAI use cases for the {industry_name} industry based on the following insights: {insights}"

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the output from the model
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)

    # Decode the output tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text
