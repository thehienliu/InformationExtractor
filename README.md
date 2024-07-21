# Information Extraction with GPT-3.5 Large Language Model

This project enables the extraction of personal information using the GPT-3.5 large language model through a web application. The application takes a name as input, utilizes SERPAPI to search for relevant information, and employs an LLM Agent to verify the collected data. The verified information is then used to generate a brief summary about the individual. The application is deployed using Streamlit for user-friendly interaction.
![example](https://github.com/user-attachments/assets/83ad9b49-3437-4a79-a1a8-228b179e3b06)

## Installation

To set up the project, follow these steps:

```bash
git clone https://github.com/thehienliu/InformationExtractor.git
cd InformationExtractor
pip install -r requirements.txt
mkdir private
```

In the private folder, create a new file api_key.py and add your API keys as follows:
```python
OPENAI_API_KEY = "<YOUR_API_KEY>"
SERPAPI_API_KEY = "<YOUR_API_KEY>"
```

## Usage

To run the Streamlit web application, use the following command:
```
streamlit run web.py
```
