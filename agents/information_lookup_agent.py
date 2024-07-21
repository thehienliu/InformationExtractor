from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from private.api_key import OPENAI_API_KEY
from langchain.agents import initialize_agent, AgentType, Tool
from tools.extractor import information_searching, image_url_searching
from output_parsers import lookup_intel_parser


def information_lookup(name: str):

    # Setup prompt template
    template = """
    Given the name {name_of_person} i want you to get the information and an image link about that person.
    Your answer should contains all information that you get and your image link should only contains the link.
    \n{format_instruction}
    """
    prompt_template = PromptTemplate(template=template, 
                                     input_variables=["name_of_person"],
                                     partial_variables={"format_instruction": lookup_intel_parser.get_format_instructions()})

    # Setup model
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY)

    # Setup tools
    tools_for_agent = [
        Tool(
            name="Searching Google for information.",
            func=information_searching,
            description="Useful when you need to get the information."
        ),
        Tool(
            name="Searching Image Link about a person.",
            func=image_url_searching,
            description="Useful when you need to get a person image link."
        )
    ]

    # Initialize agen
    agent = initialize_agent(tools=tools_for_agent,
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    
    res = agent.run(prompt_template.format_prompt(name_of_person=name))
    res = lookup_intel_parser.parse(res).to_dict()
    return res
