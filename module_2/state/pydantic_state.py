from typing import Literal
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name: str
    mood: Literal["happy", "sad"]

    @field_validator("mood")
    def validate_mood(cls, value):
        # Ensure the mood is either happy or sad
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either happy or sad")
        return value
    
try:
    state = PydanticState(name="John Doe", mood="mad")
except ValidationError as e:
    print("Validation Error:", e)