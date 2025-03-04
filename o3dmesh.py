import open3d as o3d
import torch
import numpy as np


class O3DMesh:
    def __init__(self, mesh=None, color=[0.0, 0.0, 1.0]):
        """
        Initialize Open3D mesh object.

        :param mesh: Open3D TriangleMesh object (if None, create an empty mesh)
        :param color: Default color for the mesh (RGB format)
        """
        if mesh is None:
            self.mesh = o3d.geometry.TriangleMesh()
        else:
            self.mesh = mesh

        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)

        self.vertex_normals = np.asarray(self.mesh.vertex_normals) if self.mesh.has_vertex_normals() else None
        self.face_normals = np.asarray(self.mesh.triangle_normals) if self.mesh.has_triangle_normals() else None
        self.colors = np.asarray(self.mesh.vertex_colors) if self.mesh.has_vertex_colors() else None

        if self.colors is None:
            self.set_mesh_color(color)

    def normalize_mesh(self):
        """ Normalize the mesh to fit within a unit sphere centered at the origin """
        self.mesh.normalize_normals()
        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        scale = max(bbox.extent)

        self.mesh.translate(-center)
        self.mesh.scale(1.0 / scale, center=(0, 0, 0))

        # Update numpy attributes
        self.vertices = np.asarray(self.mesh.vertices)

    def set_mesh_color(self, color):
        """
        Set uniform color for all vertices.
        :param color: RGB color as [R, G, B] in range [0,1]
        """
        color_array = np.tile(color, (len(self.vertices), 1))
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(color_array)
        self.colors = color_array

    def update_vertex_positions(self, new_vertices):
        """
        Update vertex positions of the mesh.
        :param new_vertices: New vertices as a numpy array of shape (N,3)
        """
        self.mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        self.vertices = new_vertices

    def export(self, file_path, file_format="obj"):
        """
        Export the mesh to OBJ or PLY file.
        :param file_path: Path to save the file
        :param file_format: "obj" or "ply"
        """
        if file_format.lower() == "obj":
            o3d.io.write_triangle_mesh(file_path, self.mesh, write_vertex_colors=True)
        elif file_format.lower() == "ply":
            o3d.io.write_triangle_mesh(file_path, self.mesh, write_ascii=True)
        else:
            raise ValueError("Unsupported file format. Use 'obj' or 'ply'.")

    def subdivide(self, num_iterations=1):
        """
        Subdivide the mesh using Open3D's Loop subdivision.
        :param num_iterations: Number of subdivision iterations
        """
        self.mesh = self.mesh.subdivide_loop(num_iterations)
        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)

    def compute_normals(self):
        """ Compute vertex and face normals for the mesh """
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()
        self.vertex_normals = np.asarray(self.mesh.vertex_normals)
        self.face_normals = np.asarray(self.mesh.triangle_normals)
