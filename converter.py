from torch.utils.data import Dataset
import pickle as pkl
import os.path
import torch
import numpy as np
import kaolin as kal
import open3d as o3d

affordance_descriptions = {
            "grasp": "a highlighted handle for grasping",
            "contain": "a highlighted space for containing objects",
            "lift": "a highlighted section for lifting",
            "openable": "a highlighted part that can be opened",
            "layable": "a surface for laying items",
            "sittable": "a seat for sitting",
            "support": "a structure for support",
            "wrap_grasp": "an area that can be wrap-grasped",
            "pourable": "a spout for pouring",
            "move": "a part that enables movement",
            "display": "a surface for displaying objects",
            "pushable": "a highlighted panel for pushing",
            "pull": "a handle for pulling",
            "listen": "a part for listening",
            "wear": "a wearable section",
            "press": "a button for pressing",
            "cut": "a sharp edge for cutting",
            "stab": "a pointed section for stabbing"
        }

class AffordNetDataset(Dataset):
    def __init__(self, data_dir, data):
        super().__init__()
        self.data_dir = data_dir
        self.data = data

    def load_data(self):
        self.all_data = []

        with open(os.path.join(self.data_dir, self.data), 'rb') as f:
            temp_data = pkl.load(f)

        if isinstance(temp_data, dict):
            n_arr = list(temp_data.keys())
        elif isinstance(temp_data, list):
            n_arr = list(temp_data[0].keys())

        for index, info in enumerate(temp_data):
            temp_info = {}
            temp_info["shape_id"] = info[n_arr[0]]
            temp_info["semantic_class"] = info[n_arr[1]]
            temp_info["affordance"] = info[n_arr[2]]
            temp_info["data_info"] = info[n_arr[3]]
            temp_info["coordinates"] = info[n_arr[3]]['coordinate']

            temp_info["labels"] = filter_non_zero_entries(info[n_arr[3]]['label'])
            self.all_data.append(temp_info)

        return self.all_data

def filter_non_zero_entries(input):
    # Create a new dictionary to store the filtered entries
    filtered_dict = {}
    # Iterate over each item in the input dictionary
    for key, values in input.items():
        if not np.all(values == 0):
            filtered_dict[key] = values

    return filtered_dict

def generate_clip_sentences(semantic_class, labels):
    # Generate a list of descriptions for each affordance
    if semantic_class:
        clip_texts = []
        for label in labels:
            clip_text = f"A 3D render of {semantic_class.lower()} with {affordance_descriptions.get(label, f'a highlighted {label}')}"
            clip_texts.append(clip_text)

    return clip_texts



def point_to_voxel(coordinate, resolution=64):
    if coordinate is not None:
        coordinate = (coordinate - coordinate.min()) / (coordinate.max() - coordinate.min())
        voxel_object = kal.ops.conversions.pointclouds_to_voxelgrids(pointclouds=coordinate, resolution=resolution)
    else:
        print("the coordinate is not useful")
    return voxel_object


def voxel_to_meshs(voxtel_object):

    try:
        vertices, faces = kal.ops.conversions.voxelgrids_to_trianglemeshes(voxtel_object)

        vertices = vertices[0].squeeze(0)  # (34, 3)
        faces = faces[0].squeeze(0)  # (64, 3)

        face_vertices = vertices[faces]  # (64, 3, 3)
        face_vertices = face_vertices.unsqueeze(0)

        face_normals = kal.ops.mesh.face_normals(face_vertices, unit=True)

        face_normals = face_normals.unsqueeze(3).repeat(1, 1, 1, 3)
        print(f"faces shape: {faces.shape}")
        print(f"face_normals shape: {face_normals.shape}")

        vertex_normals = kal.ops.mesh.compute_vertex_normals(faces, face_normals, len(vertices))

        vertex_normals = vertex_normals[0].squeeze(0)

        return vertices, faces, vertex_normals

    except Exception as e:
        print(f"voxel Mesh failed to create: {e}")


def point_appro_meshs(coordinate, obj_file_path, alpha=0.037):

    try:
        point_colud = o3d.geometry.PointCloud()
        point_colud.points = o3d.utility.Vector3dVector(coordinate)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_colud, alpha)
        mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(obj_file_path, mesh)
        print(f"Mesh saved at: {obj_file_path}")
    except Exception as e:
        print(f"Mesh failed to save at: {obj_file_path}: {e}")


def save_obj(filepath, vertices, vertex_normals, faces):

    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vn in vertex_normals:
            f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")

        for face in faces:
            f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")



