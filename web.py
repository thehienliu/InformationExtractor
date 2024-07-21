import streamlit as st
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from private.api_key import OPENAI_API_KEY
from langchain.prompts import PromptTemplate
from agents.information_lookup_agent import information_lookup

def setup_llm_chain(temperature: str = 0.0):

    # Setup Large Language Model
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=temperature, api_key=OPENAI_API_KEY)

    # Setup prompt template
    template = """
    Given the information {information_of_person} about a person, create me about:
        1. Summary about them
        2. 2 interesting facts about them
        3. Topic that they interesting in
        4. 2 message to open a conversation with them
    You must answer in {answer_language}.
    """
    prompt_template = PromptTemplate(template=template, input_variables=["information_of_person", "answer_language"])

    # Setup chain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain


if __name__ == "__main__":

    # Setup chain
    chain = setup_llm_chain()

    # Run streamlit web app and take input
    st.write("# Information Extractor")

    # Create form
    with st.form(key="my_form_to_submit"):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = st.text_input("Enter person name:")
        with col2:
            language = st.selectbox(label="Answer language:", options=("English", "Vietnamese"), index=0)
        submit_button = st.form_submit_button(label='Submit')

    # Run the 
    if submit_button:
        if name.strip():
            with st.spinner("Wait for it..."):
                # Get person information
                information = information_lookup(name)

                # Get final result chain
                result = chain.run(information_of_person=information, answer_language=language)
                st.write(f"## {name.title()}'s information:")
                st.write(result)