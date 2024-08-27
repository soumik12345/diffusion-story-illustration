from typing import Optional
import fire
import weave
from story_illustrator.models import StoryIllustrator


@weave.op()
def illustrate(
    story_text_path: str,
    story_title: str,
    story_author: str,
    story_setting: str,
    openai_model: str = "gpt-4",
    llm_seed: Optional[int] = None,
    diffusion_model_address: str = "black-forest-labs/FLUX.1-dev",
    enable_cpu_offoad: bool = False,
    illustration_style: Optional[str] = None,
):
    story_illustrator = StoryIllustrator(
        openai_model=openai_model,
        llm_seed=llm_seed,
        diffusion_model_address=diffusion_model_address,
        enable_cpu_offoad=enable_cpu_offoad,
    )
    with open(story_text_path, "r") as f:
        story = f.read()
    paragraphs = story.split("\n\n")
    for idx in range(10):
        story_illustrator.predict(
            story=story,
            metadata={
                "title": story_title,
                "author": story_author,
                "setting": story_setting,
            },
            paragraphs=paragraphs[idx],
            illustration_style=illustration_style,
        )


if __name__ == "__main__":
    weave.init(project_name="story-illustration")
    fire.Fire(illustrate)
