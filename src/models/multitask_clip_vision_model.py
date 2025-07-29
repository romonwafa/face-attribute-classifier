import torch.nn as nn
from transformers import CLIPVisionModel

class MultiTaskClipVisionModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        hidden_size = self.vision_model.config.hidden_size
        self.age_head = nn.Linear(hidden_size, num_labels['age'])
        self.gender_head = nn.Linear(hidden_size, num_labels['gender'])
        self.race_head = nn.Linear(hidden_size, num_labels['race'])

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return {
            'age': self.age_head(pooled_output),
            'gender': self.gender_head(pooled_output),
            'race': self.race_head(pooled_output),
        }