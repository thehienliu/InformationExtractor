from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from private.api_key import OPENAI_API_KEY
from tools.extractor import information_searching
from langchain.agents import initialize_agent, AgentType, Tool


def information_lookup(name: str):

    # Setup prompt template
    template = """
    Given the name {name_of_person} i want you to get the information about that person.
    Your answer should contains all information that you get.
    """
    prompt_template = PromptTemplate(template=template, input_variables=["name_of_person"])

    # Setup model
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY)

    # Setup tools
    tools_for_agent = [
        Tool(
            name="Searching Google for information.",
            func=information_searching,
            description="Useful when you need to get the information."
        )
    ]

    # Initialize agen
    agent = initialize_agent(tools=tools_for_agent,
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    
    res = agent.run(prompt_template.format_prompt(name_of_person=name))
    return res
