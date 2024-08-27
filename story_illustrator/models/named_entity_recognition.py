from typing import List

import weave
import instructor
from openai import OpenAI
from pydantic import BaseModel


class NameModel(BaseModel):
    character_names: List[str]


class NERModel(weave.Model):
    openai_model: str
    llm_seed: int
    _llm_client: instructor.Instructor = None

    def __init__(self, openai_model: str, llm_seed: int):
        super().__init__(openai_model=openai_model, llm_seed=llm_seed)
        self._llm_client = instructor.from_openai(OpenAI())

    @weave.op()
    def predict(self, text: str) -> NameModel:
        return self._llm_client.chat.completions.create(
            model=self.openai_model,
            response_model=NameModel,
            seed=self.llm_seed,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant meant to extract a list of names of characters from a paragraph in a story.",
                },
                {"role": "user", "content": text},
            ],
        )
