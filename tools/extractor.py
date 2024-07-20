from private.api_key import SERPAPI_API_KEY
from langchain_community.utilities import SerpAPIWrapper

def information_searching(query):
    """Search for information."""
    serpapi = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    res = serpapi.run(query)
    return res