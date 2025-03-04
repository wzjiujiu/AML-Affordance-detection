import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import kaolin as kal
from utils import device, get_camera_from_view2,get_camera_from_view1
import matplotlib.pyplot as plt


class PointCloudRenderer():
    def __init__(self, lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
                 dim=(224, 224)):

        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim

    def project_points(self, verts, camera_transform):
        """
        Projects 3D points into 2D using a (1, 4, 3) camera transformation matrix.

        Args:
            verts (torch.Tensor): 3D point cloud tensor of shape (N, 3).
            camera_transform (torch.Tensor): Camera transformation matrix of shape (1, 4, 3).

        Returns:
            projected_vertices (torch.Tensor): Projected 2D coordinates of shape (N, 2).
        """
        if camera_transform.shape != (1, 4, 3):
            raise ValueError("Camera transform should have shape (1, 4, 3).")

        # Convert verts to homogeneous coordinates (N, 3) -> (N, 4) by adding a 1 for translation
        vertices_homogeneous = torch.cat([verts, torch.ones(verts.shape[0], 1, device=verts.device)], dim=1)  # (N, 4)

        # Extract rotation (R: 3x3) and translation (t: 3x1) from the (1, 4, 3) camera matrix
        R = camera_transform[0, :3, :3]  # Rotation matrix (3x3)
        t = camera_transform[0, 3:, :3]  # Translation vector (1x3)

        # Apply rotation and translation
        transformed_vertices = torch.matmul(vertices_homogeneous[:, :3], R.T) + t  # (N, 3)

        # Perspective divide (normalize by z)
        projected_vertices = transformed_vertices[:, :2] / transformed_vertices[:, 2:].clamp(min=1e-6)  # (N, 2)

        return projected_vertices

    def render_points_as_dots(self, projected_vertices):
        # Create an empty image (background)
        image = torch.zeros(self.dim[0], self.dim[1], 3).to(device)

        # Convert the projected vertices to pixel coordinates
        projected_vertices = projected_vertices * torch.tensor([self.dim[1], self.dim[0]], device=device)
        projected_vertices = projected_vertices.long()

        # Place points as dots in the image (a small radius of 1 pixel, for instance)
        for vert in projected_vertices:
            x, y = vert[0], vert[1]
            if 0 <= x < self.dim[1] and 0 <= y < self.dim[0]:
                image[y, x] = torch.tensor([1.0, 1.0, 1.0]).to(device)  # White dot

        return image

    def render_views(self, pcd, num_views=8, std=8, center_elev=0, center_azim=0, show=False, lighting=True,
                     background=None, mask=False, return_views=False, return_mask=False):
        # Access point cloud vertices
        verts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32).to(device)  # Convert to tensor
        n_points = verts.shape[0]

        elev = torch.randn(num_views) * np.pi / std + center_elev
        azim = torch.randn(num_views) * 2 * np.pi / std + center_azim
        images = []
        masks = []

        if background is not None:
            background_mask = torch.ones(self.dim[0], self.dim[1], 3).to(device)
            background_mask *= background
        else:
            background_mask = None

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)

            # Manual Projection of vertices
            # Apply the camera transform to the vertices
            projected_vertices = self.project_points(verts, camera_transform)

            # Render the points as small dots (use Matplotlib or another method)
            image = self.render_points_as_dots(projected_vertices)
            img_np = image.cpu().numpy()  # Move to CPU & convert to NumPy
            img_np = img_np.clip(0, 1)  # Ensure values are in range [0, 1]

            # Show image
            plt.figure(figsize=(5, 5))
            plt.imshow(img_np)
            plt.axis("off")
            plt.show()

            images.append(image)


        images = torch.cat(images, dim=0).permute(0, 2, 3, 1)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views:
            return images
        else:
            return images

