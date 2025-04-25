import pickle
import os
import math

import numpy as np
import torch
from utils import device
from pathlib import Path

import random
from converter import AffordNetDataset
from converter import generate_clip_sentences, point_to_voxel, voxel_to_meshs, save_obj



def create_voxel_from_mesh(args):
    data_dir = 'data_extension'
    data_name = 'full_shape_val_data.pkl'

    affordnet = AffordNetDataset(data_dir, data_name)
    data = affordnet.load_data()
    if not data:
        raise ValueError("Loaded data is empty. Please check dataset path and contents.")

    print(data[0].keys())

    save_path = (os.path.join(data_dir, 'data_from_voxel'))
    Path(save_path).mkdir(parents=True, exist_ok=True)

    #random.seed(args.seed)
    #rand_index = random.randint(0, len(data) - 1)
    num_index = args.num_object % len(data)
    single_object = data[num_index]

    if 'semantic_class' not in single_object or 'labels' not in single_object or 'coordinates' not in single_object:
        raise KeyError("Missing required keys in data entry.")

    clip_texts = generate_clip_sentences(single_object['semantic_class'], single_object['labels'])

    coordinates = torch.tensor(single_object['coordinates']).unsqueeze(0)
    print("Coordinates Shape:", coordinates.shape)

    voxel = point_to_voxel(coordinates).to(device)
    vertices, faces, vertex_normals = voxel_to_meshs(voxel)
    #vertices = vertices[0].squeeze(0)  # shape to (34, 3)
    vertices_ = vertices.cpu().numpy()
    #faces = faces[0].squeeze(0)  # shape to  (64, 3)
    faces_ = faces.cpu().numpy()

    vertex_normals_ = vertex_normals.cpu().numpy()

    #mesh = trimesh.Trimesh(vertices=vertices_, faces=faces_)

    obj_file_path = os.path.join(save_path, f"{single_object['semantic_class']}.obj")
    #mesh.export(obj_file_path)

    save_obj(obj_file_path, vertices_, vertex_normals_, faces_)

    print(f"Mesh saved at: {obj_file_path}")

    args.object = single_object['semantic_class']
    labels = single_object['labels']

    # args.classes = single_object['labels']
    # args.prompt = clip_texts

    return args, clip_texts, obj_file_path, labels