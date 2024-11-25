import requests
from bs4 import BeautifulSoup

def fetch_industry_info(industry_name):
    """Fetch industry trends and insights using web scraping."""
    url = f"https://www.google.com/search?q={industry_name}+industry+trends+AI"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('h3')
        insights = [result.text for result in results[:5]]
        return {"industry_name": industry_name, "insights": insights}
    else:
        return {"error": "Failed to fetch industry data."}


def fetch_company_info(company_name):
    """Fetch key company offerings and focus areas."""
    url = f"https://www.google.com/search?q={company_name}+company+overview"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('h3')
        offerings = [result.text for result in results[:5]]
        return {"company_name": company_name, "offerings": offerings}
    else:
        return {"error": "Failed to fetch company data."}
