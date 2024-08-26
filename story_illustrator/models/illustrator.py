from typing import Any, Dict, List, Optional, Union

import weave
from tqdm.auto import tqdm

from .character_profiler import CharacterProfiler
from .named_entity_recognition import NERModel
from .text_to_image_generation import TextToImageGenerationModel
from .prompt_generation import InContextTextToImagePromptGenerator


class StoryIllustrator(weave.Model):
    openai_model: str
    diffusion_model_address: str
    enable_cpu_offoad: bool
    ner_model: NERModel = None
    character_profiler_model: CharacterProfiler = None
    prompt_generation_model: InContextTextToImagePromptGenerator = None
    text_to_image_model: TextToImageGenerationModel = None

    def __init__(
        self, openai_model: str, diffusion_model_address: str, enable_cpu_offoad: bool
    ):
        super().__init__(
            openai_model=openai_model,
            diffusion_model_address=diffusion_model_address,
            enable_cpu_offoad=enable_cpu_offoad,
        )
        self.ner_model = NERModel(openai_model=openai_model)
        self.character_profiler_model = CharacterProfiler(openai_model=openai_model)
        self.prompt_generation_model = InContextTextToImagePromptGenerator(
            openai_model=openai_model,
            ner_model=self.ner_model,
            character_profiler_model=self.character_profiler_model,
        )
        self.text_to_image_model = TextToImageGenerationModel(
            model_address=diffusion_model_address, enable_cpu_offoad=enable_cpu_offoad
        )

    @weave.op()
    def predict(
        self,
        story: str,
        metadata: Dict[str, Any],
        paragraphs: Optional[Union[str, List[str]]] = None,
        image_width: int = 1024,
        image_height: int = 1024,
        image_generation_guidance_scale: float = 5.0,
        image_generation_num_inference_steps: int = 28,
        image_generation_seed: Optional[int] = None,
    ) -> List[str]:
        paragraphs = [paragraphs] if isinstance(paragraphs, str) else paragraphs
        images = []
        for paragraph in tqdm(paragraphs, desc="Illustrating story"):
            summary = self.prompt_generation_model.predict(
                paragraph=paragraph,
                story=story,
                metadata=metadata,
            )
            images.append(
                self.text_to_image_model.predict(
                    prompt=summary,
                    width=image_width,
                    height=image_height,
                    guidance_scale=image_generation_guidance_scale,
                    num_inference_steps=image_generation_num_inference_steps,
                    seed=image_generation_seed,
                )
            )
        return images
