from typing import Any, Dict

import weave
from openai import OpenAI

from .named_entity_recognition import NERModel
from .character_profiler import CharacterProfiler


class TextToImagePromptGenerator(weave.Model):
    openai_model: str
    ner_model: NERModel
    character_profiler_model: CharacterProfiler
    _llm_client: OpenAI = None

    def __init__(
        self,
        openai_model: str,
        ner_model: NERModel,
        character_profiler_model: CharacterProfiler,
    ):
        super().__init__(
            openai_model=openai_model,
            ner_model=ner_model,
            character_profiler_model=character_profiler_model,
        )
        self._llm_client = OpenAI()

    @weave.op()
    def ceate_character_profiles(
        self, paragraph: str, story: str, metadata: Dict
    ) -> Dict[str, Any]:
        character_names = self.ner_model.predict(text=paragraph).character_names
        character_profiles = [
            self.character_profiler_model.predict(
                character=character_name,
                paragraph=paragraph,
                story=story,
                metadata=metadata,
            )
            for character_name in character_names
        ]
        return {
            "characters": [
                {"character_name": character_name, "profile": character_profile}
                for character_name, character_profile in zip(
                    character_names, character_profiles
                )
            ]
        }

    @weave.op()
    def predict(self, paragraph: str, story: str, metadata: Dict) -> str:
        character_profiles = self.ceate_character_profiles(
            paragraph=paragraph, story=story, metadata=metadata
        )
        system_prompt = """
You are a helpful assistant to a visionary film director. You would be provided a paragraph from a
story, the setting of the story, a profile of the characters present in the story (which includes the name
and gender of the character, how the character looks, and how they are dressed) and the story itself.
Given this information, you are supposed to summarize the paragraph in less than 40 words such that the
summary provides a detailed and accurate visual description of the paragraph which could be used by the
director and his crew to set up a scene and do a photoshoot. The summary should capture visual cues from the
time and setting of the story from the context as well as visual cues from the entire story.

Here are some rules:
1. The summary should not be less than 50 words and more than 200 words, and should not consist of more than
    2 sentences.
2. The summary should not be broken into multiple paragraphs.
3. The summary should be detailed enough to capture the visual essence of the scene described in the paragraph,
    as well as visual details of the characters present in the paragraph.
4. You should also take into account the setting of the story when generating the summary.
5. Return only the summary as the output.
6. The summary should not include the specific names of the character present in the paragraph, rather it should
    describe the characters in a general manner, describing how the gender of the character, how they look, and
    how they are dressed.
"""
        user_prompt = f"""
Here's a paragraph from a story set in {metadata['setting']}:
--------------------
{paragraph}
--------------------

Characters present in the paragraph:
--------------------
{character_profiles}
--------------------
"""

        for character_profile in character_profiles["characters"]:
            user_prompt += f"""
```
Name: {character_profile['character_name']}
Gender: {character_profile['profile'].gender}
Looks: {character_profile['profile'].looks}
Dress: {character_profile['profile'].dress}
```
"""
        user_prompt += f"""
Story:
--------------------
{story}
--------------------
        """

        return (
            self._llm_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            .choices[0]
            .message.content
        )
