import torch

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

if __name__ == "__main__":
    # # Facebook AI
    # vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    # print(vits16)

    # Hugging Face
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
    model = ViTModel.from_pretrained('facebook/dino-vits16')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(list(last_hidden_states.shape))
