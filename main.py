import argparse
from loguru import logger
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from private.api_key import OPENAI_API_KEY
from agents.information_lookup_agent import information_lookup

def parse_args():
    parser = argparse.ArgumentParser(description="Get people information.")
    parser.add_argument("--name", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="result.txt")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    information = information_lookup(args.name)
    
    template = """
    Given the information {information_of_person} about a person, create me about:
    1. Summary about them
    2. 2 interesting facts about them
    3. Topic that they interesting in
    4. 2 message to open a conversation with them
    """

    prompt_template = PromptTemplate(input_variables=["information_of_person"], template=template)
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=args.temperature, api_key=OPENAI_API_KEY)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    res = chain.run(information_of_person=information)
    logger.info("\n" + res)

    # Save output
    with open(args.output, "w+") as output_file:
        output_file.write(res)

    logger.info(f"Output saved at {args.output}!")