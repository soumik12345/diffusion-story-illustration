import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import weave
from tqdm.auto import tqdm

from .character_profiler import CharacterProfiler
from .named_entity_recognition import NERModel
from .prompt_generation import InContextTextToImagePromptGenerator
from .text_to_image_generation import DiffusersTextToImageGenerationModel


class StoryIllustrator(weave.Model):
    openai_model: str
    diffusion_model_address: str
    enable_cpu_offoad: bool
    llm_seed: Optional[int]
    ner_model: NERModel = None
    character_profiler_model: CharacterProfiler = None
    prompt_generation_model: InContextTextToImagePromptGenerator = None
    text_to_image_model: DiffusersTextToImageGenerationModel = None

    def __init__(
        self,
        openai_model: str,
        diffusion_model_address: str,
        enable_cpu_offoad: bool,
        llm_seed: Optional[int] = None,
    ):
        super().__init__(
            openai_model=openai_model,
            diffusion_model_address=diffusion_model_address,
            enable_cpu_offoad=enable_cpu_offoad,
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
        self.text_to_image_model = DiffusersTextToImageGenerationModel(
            model_address=diffusion_model_address, enable_cpu_offoad=enable_cpu_offoad
        )

    @weave.op()
    def predict(
        self,
        story: str,
        metadata: Dict[str, Any],
        paragraphs: Optional[Union[str, List[str]]] = None,
        illustration_style: Optional[str] = None,
        image_width: int = 1024,
        image_height: int = 1024,
        image_generation_guidance_scale: float = 5.0,
        image_generation_num_inference_steps: int = 28,
        use_text_encoder_2: bool = False,
        image_generation_seed: Optional[int] = None,
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
            images.append(
                self.text_to_image_model.predict(
                    prompt=summary + f" {illustration_style}",
                    width=image_width,
                    height=image_height,
                    guidance_scale=image_generation_guidance_scale,
                    num_inference_steps=image_generation_num_inference_steps,
                    use_text_encoder_2=use_text_encoder_2,
                    seed=image_generation_seed,
                )
            )
        return images
