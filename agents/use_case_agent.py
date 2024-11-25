from langchain.llms import OpenAI

def generate_use_cases(industry_name, insights):
    """Use a language model to generate AI/GenAI use cases."""
    llm = OpenAI(model="gpt-3.5-turbo", api_key="your-openai-api-key")

    prompt = f"""
    You are an AI use-case generator. Based on the following insights about the {industry_name} industry:
    {insights}

    Propose 5 innovative AI/ML/GenAI use cases that can help improve operational efficiency, customer experience, or revenue generation.
    """
    response = llm(prompt)
    return response.strip().split("\n")
