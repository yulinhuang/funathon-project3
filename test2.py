import requests
from PIL import Image

from transformers import AutoModelForSemanticSegmentation, AutoProcessor


url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoProcessor.from_pretrained("nvidia/mit-b5")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/mit-b5", device_map="auto")

inputs = processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)
logits = outputs.logits # shape [batch, num_labels, height, width]

print(logits.shape)
print(image.shape)