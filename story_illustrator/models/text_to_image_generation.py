import random
from typing import Optional, Tuple, Union

import fal_client
import numpy as np
import torch
import weave
from diffusers import DiffusionPipeline
from diffusers.utils.loading_utils import load_image
from PIL import Image

from ..utils import custom_weave_wrapper


class DiffusersTextToImageGenerationModel(weave.Model):
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
        seed = random.randint(0, np.iinfo(np.int32).max) if seed is None else seed
        return self.generate_image(
            prompt,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            use_text_encoder_2,
            seed,
        )


class FalAITextToImageGenerationModel(weave.Model):
    model_address: str

    @weave.op()
    def generate_image(
        self,
        prompt: str,
        image_size: Union[Tuple[int, int], str],
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        safety_tolerance: int,
    ) -> Image.Image:
        result = custom_weave_wrapper(name="fal_client.submit.get")(
            fal_client.submit(
                self.model_address,
                arguments={
                    "prompt": prompt,
                    "image_size": image_size,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "sync_mode": True,
                    "num_images": 1,
                    "safety_tolerance": str(safety_tolerance),
                },
            ).get
        )()
        return load_image(result["images"][0]["url"])

    @weave.op()
    def predict(
        self,
        prompt: str,
        image_size: str = "square",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        safety_tolerance: int = 2,
    ) -> Image.Image:
        assert 0 < safety_tolerance < 7
        seed = random.randint(0, np.iinfo(np.int32).max) if seed is None else seed
        return self.generate_image(
            prompt=prompt,
            image_size=image_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            safety_tolerance=safety_tolerance,
        )
