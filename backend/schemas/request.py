from pydantic import BaseModel, Field


class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=3,
        description="Question about the MiG-29 flight manual"
    )
