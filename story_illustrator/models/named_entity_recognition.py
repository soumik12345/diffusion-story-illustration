from typing import Optional

import weave
from span_marker import SpanMarkerModel


class NERModel(weave.Model):
    pretrained_model_address: str
    _pipeline: SpanMarkerModel = None
    
    def __init__(self, pretrained_model_address: str):
        super().__init__(pretrained_model_address=pretrained_model_address)
        self._pipeline = SpanMarkerModel.from_pretrained(pretrained_model_address)
    
    @weave.op()
    def predict(self, text: str) -> Optional[str]:
        return self._pipeline.predict(text)
