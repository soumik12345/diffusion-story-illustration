import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import weave
from PIL import Image
from tqdm.auto import tqdm

from .character_profiler import CharacterProfiler
from .named_entity_recognition import NERModel
from .prompt_generation import InContextTextToImagePromptGenerator
from .text_to_image_generation import (
    DiffusersTextToImageGenerationModel,
    FalAITextToImageGenerationModel,
)


class StoryIllustrator(weave.Model):
    openai_model: str
    llm_seed: Optional[int]
    text_to_image_model: Optional[weave.Model]
    ner_model: NERModel = None
    character_profiler_model: CharacterProfiler = None
    prompt_generation_model: InContextTextToImagePromptGenerator = None

    def __init__(
        self,
        openai_model: str,
        text_to_image_model: Optional[weave.Model] = None,
        llm_seed: Optional[int] = None,
    ):
        super().__init__(
            openai_model=openai_model,
            text_to_image_model=text_to_image_model,
            llm_seed=llm_seed,
        )
        self.llm_seed = (
            random.randint(0, np.iinfo(np.int32).max)
            if self.llm_seed is None
            else self.llm_seed
        )
        self.ner_model = NERModel(openai_model=openai_model, llm_seed=self.llm_seed)
        self.character_profiler_model = CharacterProfiler(
            openai_model=openai_model, llm_seed=self.llm_seed
        )
        self.prompt_generation_model = InContextTextToImagePromptGenerator(
            openai_model=openai_model,
            llm_seed=self.llm_seed,
            ner_model=self.ner_model,
            character_profiler_model=self.character_profiler_model,
        )
        self.text_to_image_model = (
            DiffusersTextToImageGenerationModel(
                model_address="black-forest-labs/FLUX.1-dev", enable_cpu_offoad=True
            )
            if self.text_to_image_model is None
            else self.text_to_image_model
        )

    @weave.op()
    def predict(
        self,
        story: str,
        metadata: Dict[str, Any],
        paragraphs: Optional[Union[str, List[str]]] = None,
        illustration_style: Optional[str] = None,
        image_size: Union[Tuple[int, int], str] = (1024, 1024),
        image_generation_guidance_scale: float = 5.0,
        image_generation_num_inference_steps: int = 28,
        use_text_encoder_2: bool = False,
        image_generation_seed: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        paragraphs = [paragraphs] if isinstance(paragraphs, str) else paragraphs
        illustration_style = "" if illustration_style is None else illustration_style
        images = []
        for paragraph in tqdm(paragraphs, desc="Illustrating story"):
            summary = self.prompt_generation_model.predict(
                paragraph=paragraph,
                story=story,
                metadata=metadata,
            )
            image: Image.Image = None
            if isinstance(
                self.text_to_image_model, DiffusersTextToImageGenerationModel
            ):
                assert isinstance(image_size, tuple)
                image_height, image_width = image_size
                image = self.text_to_image_model.predict(
                    prompt=summary + f" {illustration_style}",
                    width=image_width,
                    height=image_height,
                    guidance_scale=image_generation_guidance_scale,
                    num_inference_steps=image_generation_num_inference_steps,
                    use_text_encoder_2=use_text_encoder_2,
                    seed=image_generation_seed,
                )
            elif isinstance(self.text_to_image_model, FalAITextToImageGenerationModel):
                assert isinstance(image_size, str)
                image = self.text_to_image_model.predict(
                    prompt=summary + f" {illustration_style}",
                    image_size=image_size,
                    num_inference_steps=image_generation_num_inference_steps,
                    guidance_scale=image_generation_guidance_scale,
                    seed=image_generation_seed,
                    **kwargs,
                )
            images.append(image)
        return images
