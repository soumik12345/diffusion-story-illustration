from typing import Dict

import instructor
import weave
from pydantic import BaseModel
from openai import OpenAI


class CharacterProfile(BaseModel):
    gender: str
    looks: str
    dress: str


class CharacterProfiler(weave.Model):
    openai_model: str
    llm_seed: int
    _llm_client: instructor.Instructor = None

    def __init__(self, openai_model: str, llm_seed: int):
        super().__init__(openai_model=openai_model, llm_seed=llm_seed)
        self._llm_client = instructor.from_openai(OpenAI())

    @weave.op()
    def predict(
        self, character: str, paragraph: str, story: str, metadata: Dict
    ) -> CharacterProfile:
        return self._llm_client.chat.completions.create(
            model=self.openai_model,
            response_model=CharacterProfile,
            seed=self.llm_seed,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a expert character profiler meant to extract a character profile from a paragraph in a story.
You are to extract the gender of the character, how the character looks, and how the character dresses.

Here are some rules:
1. First look into the provided paragraph to check if it mentions the character's dress. If the paragraph
    does not mention the character's dress, you can infer the dress from the story.
2. The character's looks should be a description of the character's physical appearance. Look for clues in
    the paragraph and the story to infer the character's looks.
3. If the character's looks and dresses are not mentioned in the paragraph or the story, you should infer
    them from the setting (the time and place) of the story.
""",
                },
                {
                    "role": "user",
                    "content": f"""
Make a character profile for the character {character} in the following paragraph and the story:

Here is a paragraph from the story {metadata['title']} by {metadata['author']} and set in {metadata['setting']}.
--------------------
{paragraph}
--------------------

Story:
--------------------
{story}
--------------------
""",
                },
            ],
        )
