import open_clip
import torch
import kaolin
from mesh import Mesh
from render import Renderer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Normalization import MeshNormalizer
from torchvision import transforms

from plot_utils import highlight_mesh_portion

obj_dog = "data/dog.obj"
mesh_path = "data/dog.obj"

# highlight_mesh_portion(obj_dog,range(100))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the CLIP model names to evaluate
model_names = [
    "ViT-B/32",  # ViT-B-32 model
    "ViT-L/14",  # ViT-L-14 model
    "RN50",  # ResNet-50 model
    # Add other models if needed
]

# Load the mesh
mesh = Mesh(mesh_path)  # Load 3D mesh
render = Renderer(dim=(224, 224))
MeshNormalizer(mesh)()

# Define your text prompts for evaluation
text_prompts = [
    "A 3D render of blue dog "
]

# Tokenize the text prompts
text_tokens = open_clip.tokenize(text_prompts).to(device)



background = (1.0, 1.0, 1.0)
background_tensor = torch.tensor(background, dtype=torch.float32, device=device).view(1, 1, 1, 3)

# Store the best model and similarity score
best_model = None
best_similarity = -float('inf')
best_model_name = ""

# Iterate over all CLIP models and compute similarities
for model_name in model_names:
    print(f"Evaluating model: {model_name}")

    # Load the CLIP model and preprocessing pipeline
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai", device=device)

    # Perform CLIP inference
    with torch.no_grad():
        # Render the 3D mesh
        rendered_images, elev, azim = render.render_views(
            mesh, num_views=20, show='store_true',
            center_azim=0., center_elev=0.,
            std=4, return_views=True, lighting=True, background=background_tensor
        )

        # Store similarity scores for the current model
        model_similarity_scores = []

        for i, image_tensor in enumerate(rendered_images):
            # Convert tensor to PIL image for preprocessing
            image_pil = Image.fromarray(
                (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))  # Convert tensor to PIL image

            # Preprocess the rendered image for CLIP
            clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))
            clip_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                clip_normalizer
            ])

            image = clip_transform(image_pil)
            image = image.unsqueeze(0).to(device)

            # Compute image embedding
            image_features = model.encode_image(image)

            # Normalize the image features
            image_features /= image_features.norm(dim=1, keepdim=True)

            # Compute text embedding
            text_features = model.encode_text(text_tokens)

            # Normalize the text features
            text_features /= text_features.norm(dim=1, keepdim=True)

            # Compute cosine similarity between image and text
            similarity = (image_features @ text_features.T).squeeze()

            # Append similarity score for this image
            model_similarity_scores.append(similarity.item())

            # Optionally print the similarity for each image
            print(f"Similarity with '{text_prompts[0]}' for image {i + 1}: {similarity.item():.4f}")

        # Compute the average similarity for the current model
        avg_similarity = np.mean(model_similarity_scores)
        print(f"Average similarity for model {model_name}: {avg_similarity:.4f}")

        # Update the best model if the current one has a higher average similarity
        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_model_name = model_name
            best_model = model

# Output the best model and its similarity
print(f"Best model: {best_model_name} with average similarity: {best_similarity:.4f}")