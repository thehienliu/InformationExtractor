from private.api_key import SERPAPI_API_KEY
from langchain_community.utilities import SerpAPIWrapper

class CustomSerpAPIWrapper(SerpAPIWrapper):
    def __init__(self, serpapi_api_key):
        super().__init__(serpapi_api_key=serpapi_api_key)
        self.params["engine"] = "google_images"

    @staticmethod
    def _process_response(res: dict) -> str:
        return res["images_results"][0]["original"]

def information_searching(query):
    """Search for information."""
    serpapi = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    res = serpapi.run(query)
    return res


def image_url_searching(query):
    """Search for person image."""
    serpapi = CustomSerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    res = serpapi.run(query)
    return res




