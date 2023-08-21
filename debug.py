import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print ('preprocess', preprocess)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
print ( 'image shape',  image.shape)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
print( 'text shape:', text.shape)

with torch.no_grad():
    image_features = model.encode_image(image)
    print( 'image_features shape:', image_features.shape)

    text_features = model.encode_text(text)
    print( 'text_features shape:', text_features.shape)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
