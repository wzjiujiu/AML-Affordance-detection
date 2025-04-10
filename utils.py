import torch
import kaolin as kal
import kaolin.ops.mesh
import clip
import numpy as np
from torchvision import transforms
from pathlib import Path
from collections import Counter
from Normalization import MeshNormalizer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj

def get_camera_from_view1(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)

    pos = torch.tensor([x, y, z]).unsqueeze(0).to(device)  # Camera position
    look_at = -pos  # Look at the origin
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(device)  # Up direction

    # Generate a (1, 4, 3) transformation matrix
    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)  # Shape (1, 4, 3)

    # Ensure it's (4, 4) by adding [0, 0, 0, 1] row
    if camera_proj.shape == (1, 4, 3):
        last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(camera_proj.device)  # Ensure same device
        camera_proj = torch.cat([camera_proj.squeeze(0), last_row], dim=0)  # Shape now (4, 4)

    return camera_proj

def get_texture_map_from_color(mesh, color, H=224, W=224):
    num_faces = mesh.faces.shape[0]
    texture_map = torch.zeros(1, H, W, 3).to(device)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)
    face_attributes[:, :, :] = color
    return face_attributes


# ================== POSITIONAL ENCODERS =============================
# using fourier transformation increase data  detail for network
class FourierFeatureTransform(torch.nn.Module):
    def __init__(self, input_dim, output_dim=256, sigma=5):
        super(FourierFeatureTransform, self).__init__()
        self.sigma = sigma
        self.B = torch.randn((input_dim, output_dim)) * sigma

    def forward(self, x):
        res = x @ self.B
        x_sin = torch.sin(res)
        x_cos = torch.cos(res)

        return torch.cat([x, x_sin, x_cos], dim=1)


# mesh coloring helpers
def color_mesh(pred_class, sampled_mesh, colors):
    pred_rgb = segment2rgb(pred_class, colors)
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    MeshNormalizer(sampled_mesh)()

def segment2rgb(pred_class, colors):
    pred_rgb = torch.zeros(pred_class.shape[0], 3).to(device)
    for class_idx, color in enumerate(colors):
        pred_rgb += torch.matmul(pred_class[:,class_idx].unsqueeze(1), color.unsqueeze(0))
        
    return pred_rgb

def compute_miou(pred_mask, gt_mask):

    assert pred_mask.shape[0] == gt_mask.shape[0]

    # Ensure gt_mask is 1D if it has an extra dimension
    if gt_mask.ndim == 2 and gt_mask.shape[1] == 1:
        gt_mask = gt_mask.squeeze(1)  # Convert from (N,1) → (N,)

    gt_mask=(gt_mask != 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    return intersection / union if union > 0 else 0.0  # Avoid division by zero