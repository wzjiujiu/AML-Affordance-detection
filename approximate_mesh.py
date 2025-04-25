from pathlib import Path

import os
import torch
import random
from converter import AffordNetDataset
from converter import generate_clip_sentences, point_appro_meshs


def create_appro_mesh(args):
    data_dir = 'data_extension'
    data_name = 'full_shape_val_data.pkl'

    affordnet = AffordNetDataset(data_dir, data_name)
    data = affordnet.load_data()
    if not data:
        raise ValueError("Loaded data is empty. Please check dataset path and contents.")

    print(data[0].keys())

    save_path = (os.path.join(data_dir, 'data_from_appro'))
    Path(save_path).mkdir(parents=True, exist_ok=True)

    #random.seed(args.seed)
    #rand_index = random.randint(0, len(data) - 1)
    num_index = args.num_object % len(data)
    single_object = data[num_index]

    if 'semantic_class' not in single_object or 'labels' not in single_object or 'coordinates' not in single_object:
        raise KeyError("Missing required keys in data entry.")

    clip_texts = generate_clip_sentences(single_object['semantic_class'], single_object['labels'])

    coordinates = torch.tensor(single_object['coordinates'])
    print("Coordinates Shape:", coordinates.shape)

    obj_file_path = os.path.join(save_path, f"{single_object['semantic_class']}.obj")

    point_appro_meshs(coordinates, obj_file_path)
    args.object = single_object['semantic_class']
    labels = single_object['labels']

    # args.classes = single_object['labels']
    # args.prompt = clip_texts

    return args, clip_texts, obj_file_path, labels
