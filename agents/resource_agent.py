import requests

def search_datasets(use_case):
    """Search for datasets on Kaggle or HuggingFace."""
    search_url = f"https://www.kaggle.com/search?q={use_case.replace(' ', '+')}"
    response = requests.get(search_url)

    if response.status_code == 200:
        return f"Kaggle search link for '{use_case}': {search_url}"
    else:
        return "No datasets found for this use case."

def collect_resources(use_cases):
    """Collect resource links for each use case."""
    resources = {}
    for use_case in use_cases:
        resources[use_case] = search_datasets(use_case)
    return resources
