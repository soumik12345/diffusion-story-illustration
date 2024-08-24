import os
import json
from typing import Dict, Optional

import weave
from openai import OpenAI

from .named_entity_recognition import NERModel


class DiffusionPromptGenerator(weave.Model):
    openai_model: str
    ner_model: NERModel
    _openai_client: Optional[OpenAI] = None
    
    def __init__(self, openai_model: str, ner_model: NERModel):
        super().__init__(openai_model=openai_model, ner_model=ner_model)
        self._openai_client = OpenAI()
    
    @weave.op()
    def predict(self, paragraph: str, story: str, metadata: Dict) -> str:
        named_entities = self.ner_model.predict(text=paragraph)
        person_names = [entity["span"] for entity in named_entities if entity["label"] == "PER"]
