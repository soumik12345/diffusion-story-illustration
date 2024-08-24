# Diffusion Story Illustration

## Sample Workflow

```python
import json
import weave
from story_illustrator.models import (
    CharacterProfiler,
    DiffusionPromptGenerator,
    NERModel,
)


weave.init(project_name="story-illustration")

ner_model = NERModel(openai_model="gpt-4")
character_profiler_model = CharacterProfiler(openai_model="gpt-4")
prompt_generation_model = DiffusionPromptGenerator(
    openai_model="gpt-4",
    ner_model=ner_model,
    character_profiler_model=character_profiler_model,
)

with open("./data/gift_of_the_magi/story.txt", "r") as f:
    story = f.read()
with open("./data/gift_of_the_magi/metadata.json", "r") as f:
    metadata = json.load(f)
enities = prompt_generation_model.predict(
    paragraph="""ONE DOLLAR AND eighty-seven cents. That was all. And sixty cents of it was in pennies. Pennies saved one and two at a time by bulldozing the grocer and the vegetable man and the butcher until one's cheeks burned with the silent imputation of parsimony that such close dealing implied. Three times Della counted it. One dollar and eighty-seven cents. And the next day would be Christmas.""",
    story=story,
    metadata=metadata,
)

```