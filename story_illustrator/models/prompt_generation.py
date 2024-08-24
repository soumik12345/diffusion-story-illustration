from typing import Any, Dict, Optional

import weave
from openai import OpenAI

from .named_entity_recognition import NERModel
from .character_profiler import CharacterProfiler


class DiffusionPromptGenerator(weave.Model):
    openai_model: str
    ner_model: NERModel
    character_profiler_model: CharacterProfiler
    _openai_client: Optional[OpenAI] = None

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

    @weave.op()
    def predict(self, paragraph: str, story: str, metadata: Dict) -> Dict[str, Any]:
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
        final_response = {
            "characters": [
                {"character_name": character_name, "profile": character_profile}
                for character_name, character_profile in zip(
                    character_names, character_profiles
                )
            ]
        }
        return final_response
