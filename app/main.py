import streamlit as st
from agents.research_agent import fetch_industry_info, fetch_company_info
from agents.use_case_agent import generate_use_cases
from agents.resource_agent import collect_resources

from agents.use_case_agent import generate_use_cases
from agents.resource_agent import collect_resources

st.title("AI/GenAI Market Research & Use Case Generator")

# Input fields
industry_name = st.text_input("Enter the Industry Name (e.g., Retail, Healthcare):")
company_name = st.text_input("Enter the Company Name (optional):")

if st.button("Generate Insights"):
    # Fetch industry and company data
    industry_info = fetch_industry_info(industry_name)
    st.subheader("Industry Insights")
    st.write(industry_info)

    if company_name:
        company_info = fetch_company_info(company_name)
        st.subheader("Company Offerings")
        st.write(company_info)

    # Generate AI use cases
    use_cases = generate_use_cases(industry_name, industry_info['insights'])
    st.subheader("AI/GenAI Use Cases")
    for use_case in use_cases:
        st.write(f"- {use_case}")

    # Collect resources
    st.subheader("Relevant Datasets & Tools")
    resources = collect_resources(use_cases)
    for use_case, link in resources.items():
        st.write(f"{use_case}: [Dataset Link]({link})")
