import kaolin as kal
import torch
import utils
from utils import device
import copy
import numpy as np
import PIL
import open3d as o3d

class Mesh():
    def __init__(self,input_data,color=torch.tensor([0.0,0.0,1.0])):
        """
                Initialize Mesh from Open3D TriangleMesh object.

                Args:
                    mesh (open3d.geometry.TriangleMesh): Input Open3D mesh.
                    color (torch.tensor, optional): Color for the mesh. Defaults to blue.
                """
        if isinstance(input_data, o3d.geometry.TriangleMesh):
            # Initialize from Open3D TriangleMesh object
            self.vertices = torch.tensor(np.asarray(input_data.vertices), dtype=torch.float32).to(device)
            self.faces = torch.tensor(np.asarray(input_data.triangles), dtype=torch.int64).to(device)
            self.vertex_normals = None
            self.face_normals = None
            self.texture_map = None
            self.face_uvs = None

            # Handle vertex normals if available
            if input_data.has_vertex_normals():
                self.vertex_normals = torch.tensor(np.asarray(input_data.vertex_normals), dtype=torch.float32).to(
                    device)
                # Normalize the normals
                self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

            # Handle face normals if available
            if input_data.has_triangle_normals():
                self.face_normals = torch.tensor(np.asarray(input_data.triangle_normals), dtype=torch.float32).to(
                    device)
                # Normalize the face normals
                self.face_normals = torch.nn.functional.normalize(self.face_normals)

            # Optionally handle texture maps or colors
            self.set_mesh_color(color)

        elif isinstance(input_data, str):
            # Initialize from file path (obj or off)
            file_path = input_data.lower()
            if file_path.endswith(".obj"):
                # Load .obj file using Kaolin (or similar)
                mesh = kal.io.obj.import_mesh(file_path, with_normals=True)
                self.vertices = mesh.vertices.to(device)
                self.faces = mesh.faces.to(device)
                self.vertex_normals = None
                self.face_normals = None
                self.texture_map = None
                if mesh.vertex_normals is not None:
                    self.vertex_normals = mesh.vertex_normals.to(device).float()
                    # Normalize
                    self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

                if mesh.face_normals is not None:
                    self.face_normals = mesh.face_normals.to(device).float()
                    # Normalize
                    self.face_normals = torch.nn.functional.normalize(self.face_normals)

                # Handle color
                self.set_mesh_color(color)

            elif file_path.endswith(".off"):
                # Load .off file using Kaolin (or similar)
                mesh = kal.io.off.import_mesh(file_path)
                self.vertices = mesh.vertices.to(device)
                self.faces = mesh.faces.to(device)
                self.vertex_normals = None
                self.face_normals = None
                self.texture_map = None
                self.set_mesh_color(color)

            else:
                raise ValueError(f"Unsupported file extension in {file_path}")

        else:
            raise ValueError("Input data must be either an Open3D TriangleMesh or a file path (obj/off).")

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return utils.standardize_mesh(mesh)

    def normalize_mesh(self,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        return utils.normalize_mesh(mesh)

    def update_vertex(self,verts,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self,color):
        self.texture_map = utils.get_texture_map_from_color(self,color)
        self.face_attributes = utils.get_face_attributes_from_color(self,color)

    def set_image_texture(self,texture_map,inplace=True):

        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map,str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map,dtype=np.float) / 255.0
            texture_map = torch.tensor(texture_map,dtype=torch.float).to(device).permute(2,0,1).unsqueeze(0)


        mesh.texture_map = texture_map
        return mesh

    def divide(self,inplace=True):
        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_vertex_normals, new_face_uvs, new_face_normals = utils.add_vertices(mesh)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.vertex_normals = new_vertex_normals
        mesh.face_uvs = new_face_uvs
        mesh.face_normals = new_face_normals
        return mesh

    def divide_with_labels(self, face_label_map, inplace=True):
        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_vertex_normals, new_face_label_map = utils.add_vertices_with_labels(mesh, face_label_map)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.vertex_normals = new_vertex_normals
        return mesh, new_face_label_map

    def export(self, file, extension="obj", color=None):
        with open(file, "w+") as f:
            if extension == "obj":
                for vi, v in enumerate(self.vertices):
                    if color is None:
                        f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                    else:
                        f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                    if self.vertex_normals is not None:
                        f.write("vn %f %f %f\n" % (self.vertex_normals[vi, 0], self.vertex_normals[vi, 1], self.vertex_normals[vi, 2]))
                for face in self.faces:
                    f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))
            elif extension == "ply":
                f.write("ply\nformat ascii 1.0\n")
                f.write("element vertex {}\n".format(self.vertices.shape[0]))
                f.write("property float32 x\nproperty float32 y\nproperty float32 z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("element face {}\n".format(self.faces.shape[0]))
                f.write("property list uint8 int32 vertex_index\n")
                f.write("end_header\n")
                for vi, v in enumerate(self.vertices):
                    f.write("%f %f %f %d %d %d\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                for face in self.faces:
                    f.write("3 %d %d %d\n" % (face[0], face[1], face[2]))
            else:
                print("error -- mesh.py: incorrect extension provdied in mesh.export()")
