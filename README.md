# Diffusion Story Illustration [WIP]

## Sample Workflow

```python
import weave
from story_illustrator.models import StoryIllustrator


weave.init(project_name="story-illustration")

story_illustrator = StoryIllustrator(
    openai_model="gpt-4",
    diffusion_model_address="black-forest-labs/FLUX.1-dev",
    enable_cpu_offoad=False,
)
with open("./data/gift_of_the_magi.txt", "r") as f:
    story = f.read()
paragraphs = story.split("\n\n")
story_illustrator.predict(
    story=story,
    metadata={
        "title": "Gift of the Magi",
        "author": "O. Henry",
        "setting": "the year 1905, New York City, United States of America",
    },
    paragraphs=paragraphs[:2],
)
```