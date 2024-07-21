from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class PersonIntel(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: List[str] = Field(description="Interesting facts about the person")
    topics_of_interest: str = Field(description="Topic that may interest the person")
    ice_breakers: List[str] = Field(description="Create ice breakers to opn a conversation with the person")

    def to_dict(self):
        return {"summary": self.summary, "facts": self.facts, "topics_of_interest": self.topics_of_interest, "ice_breakers": self.ice_breakers}

class LookupIntel(BaseModel):
    image_link: str = Field(description="Image link about the person")
    information: str = Field(description="All the observation information about the person")

    def to_dict(self):
        return {"information": self.information, "image_link": self.image_link}
    
person_intel_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=PersonIntel)
lookup_intel_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=LookupIntel)