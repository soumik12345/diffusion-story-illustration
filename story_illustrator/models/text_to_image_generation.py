import random
from typing import Optional

import numpy as np
import torch
import weave
from diffusers import DiffusionPipeline
from PIL import Image


class TextToImageGenerationModel(weave.Model):
    model_address: str
    enable_cpu_offoad: bool
    _pipeline: DiffusionPipeline = None

    def __init__(self, model_address: str, enable_cpu_offoad: bool):
        super().__init__(
            model_address=model_address, enable_cpu_offoad=enable_cpu_offoad
        )
        self._pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_address, torch_dtype=torch.bfloat16
        )
        if self.enable_cpu_offoad:
            self._pipeline.enable_model_cpu_offload()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = self._pipeline.to(device)
        self._pipeline.set_progress_bar_config(leave=False, desc="Generating Image")

    @weave.op()
    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
        use_text_encoder_2: bool = False,
        seed: Optional[int] = None,
    ) -> Image.Image:
        generator = torch.Generator().manual_seed(seed)
        return self._pipeline(
            prompt="" if use_text_encoder_2 else prompt,
            prompt_2=prompt if use_text_encoder_2 else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

    @weave.op()
    def predict(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 28,
        use_text_encoder_2: bool = False,
        seed: Optional[int] = None,
    ) -> Image.Image:
        seed = random.randint(0, np.iinfo(np.int32).max)
        return self.generate_image(
            prompt,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            use_text_encoder_2,
            seed,
        )
