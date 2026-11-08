from pydantic import BaseModel, Field


class TextPayload(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    prediction: int
    flagged_probability: float = Field(ge=0.0, le=1.0)
    status: str
