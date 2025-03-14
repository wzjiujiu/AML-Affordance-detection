import clip
import copy
import json
import kaolin as kal
import numpy as np
import os
import random
import torch
import sys
import torch.nn as nn
import torchvision
from neural_highlighter import NeuralHighlighter
from Normalization import MeshNormalizer
from mesh import Mesh
from pathlib import Path
from render import Renderer
from tqdm import tqdm
from torchvision import transforms
from utils import device, color_mesh

from argparse import ArgumentParser
import open3d as o3d
from AffoLoad import PklLoader
from converter import AffordNetDataset, generate_clip_sentences, point_to_voxel, voxel_to_meshs
from voxel_mesh import create_voxel_from_mesh, voxel_from_mesh



def obj_to_point_cloud(obj_path, ply_path):
    points = []

    # Read the OBJ file
    with open(obj_path, 'r') as file:
        for line in file:
            if line.startswith("v "):  # Only extract vertex positions
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])  # Extract XYZ coordinates
                points.append([x, y, z])

    # Convert to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Save as a PLY file
    o3d.io.write_point_cloud(ply_path, point_cloud)
    print(f"Point cloud saved to {ply_path}")


def extract_vertices_from_ply(ply_file, device="cpu", alpha=0.03):
    """Extracts vertex positions from a PLY file and moves them to the specified device."""
    pcd = o3d.io.read_point_cloud(ply_file)  # Load point cloud
    sampled_pcd=pcd.uniform_down_sample(every_k_points=100)
    vertices = np.asarray(pcd.points)  # Extract vertex coordinates


    # Generate mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)


    return pcd, torch.tensor(vertices, dtype=torch.float32, device=device), mesh  # Move tensor to device


def optimize(args):
    # Constrain most sources of randomness
    # (some torch backwards functions within CLIP are non-determinstic)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)

    # Adjust output resolution depending on model type
    res = 224
    if args.clipmodel == "ViT-L/14@336px":
        res = 336
    if args.clipmodel == "RN50x4":
        res = 288
    if args.clipmodel == "RN50x16":
        res = 384
    if args.clipmodel == "RN50x64":
        res = 448

    Path(os.path.join(args.output_dir, 'renders')).mkdir(parents=True, exist_ok=True)

    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))

    render = Renderer(dim=(args.render_res, args.render_res))
    mesh = Mesh(args.obj_path)
    # load the mesh  加载mesh
    MeshNormalizer(mesh)()  # Normalize the mesh to fit within a consistent space

    # Initialize variables
    background = None
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)
    n_augs = args.n_augs
    dir = args.output_dir

    # Record command line arguments
    with open(os.path.join(dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # CLIP and Augmentation Transforms
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_transform = transforms.Compose([
        transforms.Resize((res, res)),
        #clip_normalizer,
        preprocess
    ])
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])

    # MLP Settings
    # 高亮模型初始化
    mlp = NeuralHighlighter(args.depth, args.width, out_dim=args.n_classes,
                            positional_encoding=args.positional_encoding,
                            sigma=args.sigma).to(device)

    # explanation : 1. n_calsses (since there are two colors: "highlighter" and "gray").
    # num_vertices = number of vertices in the 3D mesh.

    optim = torch.optim.Adam(mlp.parameters(), args.learning_rate)

    # list of possible colors
    rgb_to_color = {(204 / 255, 1., 0.): "highlighter", (180 / 255, 180 / 255, 180 / 255): "gray"}
    color_to_rgb = {"highlighter": [204 / 255, 1., 0.], "gray": [180 / 255, 180 / 255, 180 / 255]}
    full_colors = [[204 / 255, 1., 0.], [180 / 255, 180 / 255, 180 / 255]]  # first is yellow, second is gray
    colors = torch.tensor(full_colors).to(device)

    # --- Prompt ---
    # pre-process multi_word_inputs
    # encode prompt with CLIP
    #prompt = "A 3D render of a gray hat with a highlighted candle."
    prompt = args.prompt
    with torch.no_grad():
        prompt_token = clip.tokenize([prompt]).to(device)
        encoded_text = clip_model.encode_text(prompt_token)
        encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)

    if args.point_cloud:
        mlp.to(dev)
        vertices = extract_vertices_from_ply('candle.ply', device=dev)
    else:
        vertices = copy.deepcopy(mesh.vertices)

    losses = []

    # Optimization loop
    for i in tqdm(range(args.n_iter)):
        optim.zero_grad()

        # predict highlight probabilities
        pred_class = mlp(vertices)

        # color and render mesh
        sampled_mesh = mesh
        color_mesh(pred_class, sampled_mesh, colors)
        rendered_images, elev, azim = render.render_views(sampled_mesh, num_views=args.n_views,
                                                          show=False,
                                                          center_azim=args.frontview_center[0],
                                                          center_elev=args.frontview_center[1],
                                                          std=args.frontview_std,
                                                          return_views=True,
                                                          lighting=True,
                                                          background=background)

        # Calculate CLIP Loss
        # 计算CLIP lOSS
        loss = clip_loss(args, rendered_images, encoded_text, clip_transform, augment_transform, clip_model)

        #
        loss.backward(retain_graph=True)

        optim.step()

        # update variables + record loss
        with torch.no_grad():
            losses.append(loss.item())

        # report results
        if i % 100 == 0:
            render.render_views(sampled_mesh, num_views=args.n_views,
                                show=True,
                                center_azim=args.frontview_center[0],
                                center_elev=args.frontview_center[1],
                                std=args.frontview_std,
                                return_views=False,
                                lighting=True,
                                background=background)

            print("Last 100 CLIP score: {}".format(np.mean(losses[-100:])))
            save_renders(dir, i, rendered_images)
            with open(os.path.join(dir, "training_info.txt"), "a") as f:
                f.write(
                    f"For iteration {i}... Prompt: {prompt}, Last 100 avg CLIP score: {np.mean(losses[-100:])}, CLIP score {losses[-1]}\n")

    # re-initialize background color
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)
    # save results
    save_final_results(args, dir, mesh, mlp, vertices, colors, render, background)

    # Save prompts
    with open(os.path.join(dir, prompt), "w") as f:
        f.write('')


# ================== HELPER FUNCTIONS =============================
def save_final_results(args, dir, mesh, mlp, vertices, colors, render, background):
    mlp.eval()
    with torch.no_grad():
        probs = mlp(vertices)
        max_idx = torch.argmax(probs, 1, keepdim=True)
        # for renders
        one_hot = torch.zeros(probs.shape).to(device)
        one_hot = one_hot.scatter_(1, max_idx, 1)
        sampled_mesh = mesh

        highlight = torch.tensor([204, 255, 0]).to(device)
        gray = torch.tensor([180, 180, 180]).to(device)
        colors = torch.stack((highlight / 255, gray / 255)).to(device)
        color_mesh(one_hot, sampled_mesh, colors)
        rendered_images, _, _ = render.render_views(sampled_mesh, num_views=args.n_views,
                                                    show=args.show,
                                                    center_azim=args.frontview_center[0],
                                                    center_elev=args.frontview_center[1],
                                                    std=args.frontview_std,
                                                    return_views=True,
                                                    lighting=True,
                                                    background=background)
        # for mesh
        final_color = torch.zeros(vertices.shape[0], 3).to(device)
        final_color = torch.where(max_idx == 0, highlight, gray)
        objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        mesh.export(os.path.join(dir, f"{objbase}_{args.classes[0]}.ply"), extension="ply", color=final_color)
        save_renders(dir, 0, rendered_images, name='final_render.jpg')


def clip_loss(args, rendered_images, encoded_text, clip_transform, augment_transform, clip_model):
    # here here loss should be small because of minus
    if args.n_augs == 0:
        loss = 0.0
        clip_image = clip_transform(rendered_images)
        encoded_renders = clip_model.encode_image(clip_image)
        encoded_renders = encoded_renders / encoded_renders.norm(dim=1, keepdim=True)
        if args.clipavg == "view":
            if encoded_text.shape[0] > 1:
                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                               torch.mean(encoded_text, dim=0), dim=0)
            else:
                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                               encoded_text)
        else:
            loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
    # here loss should be small because of minus and augmentation
    elif args.n_augs > 0:
        loss = 0.0
        for _ in range(args.n_augs):
            augmented_image = augment_transform(rendered_images)
            encoded_renders = clip_model.encode_image(augmented_image)
            if args.clipavg == "view":
                if encoded_text.shape[0] > 1:
                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                    torch.mean(encoded_text, dim=0), dim=0)
                else:
                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                    encoded_text)
            else:
                loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
    return loss


def save_renders(dir, i, rendered_images, name=None):
    if name is not None:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, name))
    else:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, 'renders/iter_{}.jpg'.format(i)))


def optimize_affonet(agrs, clip_sentence, filepath, gt_label,obj):
    # Constrain most sources of randomness
    # (some torch backwards functions within CLIP are non-determinstic)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render = Renderer(dim=(args.render_res, args.render_res))

    # Load CLIP model
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)

    # Adjust output resolution depending on model type
    res = 224
    if args.clipmodel == "ViT-L/14@336px":
        res = 336
    if args.clipmodel == "RN50x4":
        res = 288
    if args.clipmodel == "RN50x16":
        res = 384
    if args.clipmodel == "RN50x64":
        res = 448

    Path(os.path.join(args.output_dir, 'renders')).mkdir(parents=True, exist_ok=True)


    # Initialize variables
    background = None
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)
    n_augs = args.n_augs
    dir = args.output_dir

    # CLIP and Augmentation Transforms
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_transform = transforms.Compose([
        transforms.Resize((res, res)),
        clip_normalizer
    ])
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])

    # MLP Settings
    # 高亮模型初始化
    mlp = NeuralHighlighter(args.depth, args.width, out_dim=args.n_classes,
                            positional_encoding=args.positional_encoding,
                            sigma=args.sigma).to(device)

    # explanation : 1. n_calsses (since there are two colors: "highlighter" and "gray").
    # num_vertices = number of vertices in the 3D mesh.

    optim = torch.optim.Adam(mlp.parameters(), args.learning_rate)

    # list of possible colors
    rgb_to_color = {(204 / 255, 1., 0.): "highlighter", (180 / 255, 180 / 255, 180 / 255): "gray"}
    color_to_rgb = {"highlighter": [204 / 255, 1., 0.], "gray": [180 / 255, 180 / 255, 180 / 255]}
    full_colors = [[204 / 255, 1., 0.], [180 / 255, 180 / 255, 180 / 255]]  # first is yellow, second is gray
    colors = torch.tensor(full_colors).to(device)

    # --- Prompt ---
    # pre-process multi_word_inputs
    # encode prompt with CLIP
    prompt = clip_sentence
    with torch.no_grad():
        prompt_token = clip.tokenize([prompt]).to(device)
        encoded_text = clip_model.encode_text(prompt_token)
        encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)

    mlp.to(dev)
    point_cloud, vertices, mesh_ = extract_vertices_from_ply(filepath, device=dev, alpha=0.03)

    mesh = Mesh(mesh_)

    # load the mesh  加载mesh
    MeshNormalizer(mesh)()  # Normalize the mesh to fit within a consistent space

    #  This is a tensor of shape (num_vertices, 3), where each row represents the (x, y, z) coordinates of a vertex.
    # 网格图所有坐标点提取

    losses = []

    # Optimization loop
    for i in tqdm(range(args.n_iter)):
        optim.zero_grad()

        # predict highlight probabilities
        pred_class = mlp(vertices)

        # color and render mesh
        # 图形渲染

        sampled_mesh = mesh
        color_mesh(pred_class, sampled_mesh, colors)
        rendered_images, elev, azim = render.render_views(sampled_mesh, num_views=args.n_views,
                                                          show=False,
                                                          center_azim=args.frontview_center[0],
                                                          center_elev=args.frontview_center[1],
                                                          std=args.frontview_std,
                                                          return_views=True,
                                                          lighting=True,
                                                          background=background)

        # Calculate CLIP Loss
        # 计算CLIP lOSS
        loss = clip_loss(args, rendered_images, encoded_text, clip_transform, augment_transform, clip_model)

        #
        loss.backward(retain_graph=True)

        optim.step()

        # update variables + record loss
        with torch.no_grad():
            losses.append(loss.item())

        # report results
        if i % 100 == 0:
            render.render_views(sampled_mesh, num_views=args.n_views,
                                show=True,
                                center_azim=args.frontview_center[0],
                                center_elev=args.frontview_center[1],
                                std=args.frontview_std,
                                return_views=False,
                                lighting=True,
                                background=background)

            print("Last 100 CLIP score: {}".format(np.mean(losses[-100:])))
            save_renders(dir, i, rendered_images)
            with open(os.path.join(dir, "training_info.txt"), "a") as f:
                f.write(
                    f"For iteration {i}... Prompt: {prompt}, Last 100 avg CLIP score: {np.mean(losses[-100:])}, CLIP score {losses[-1]}\n")

    # re-initialize background color
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)
    # save results
    save_final_results(args, dir, mesh, mlp, vertices, colors, render, background)

    # Save prompts
    with open(os.path.join(dir, prompt), "w") as f:
        f.write('')


def compute_mIOU(gt_mask, pred_mask, num_classes):
    """
    Compute the mean Intersection over Union (mIOU) score.

    Args:
        gt_mask (np.array): Ground truth segmentation mask (H, W) with class labels.
        pred_mask (np.array): Predicted segmentation mask (H, W) with class labels.
        num_classes (int): Total number of affordance classes.

    Returns:
        float: mIOU score.
    """
    ious = []
    for cls in range(num_classes):
        gt_cls = (gt_mask == cls)
        pred_cls = (pred_mask == cls)

        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()

        if union == 0:
            ious.append(1.0)  # If there's no GT and no prediction, it's a perfect match
        else:
            ious.append(intersection / union)

    return np.mean(ious)

if __name__ == '__main__':

    # Define your parser
    parser = ArgumentParser()

    # Add arguments as before
    parser.add_argument('--voxel', type=str, default=True)   #using voxel
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obj_path', type=str, default='data/candle.obj')
    parser.add_argument('--output_dir', type=str, default='results/segment/1')
    parser.add_argument('--prompt', nargs="+", default='a pig with pants')
    parser.add_argument('--object', nargs=1, default='cow')
    parser.add_argument('--classes', nargs="+", default='sphere cube')
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--n_augs', type=int, default=1)
    parser.add_argument('--clipavg', type=str, default='view')
    parser.add_argument('--render_res', type=int, default=224)
    parser.add_argument('--clipmodel', type=str, default='ViT-L/14')
    parser.add_argument('--jit', action="store_true")
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_iter', type=int, default=2500)
    parser.add_argument('--point_cloud', action='store_false', help="Disable point cloud mode (default: True)")

    args = parser.parse_args()

    #data load
    path = 'point_clouds/'
    Affloader = PklLoader("AffoNet/full_shape_val_data.pkl")

    Affloader.load_data()

    data = Affloader.get_data()
    shape_ids = Affloader.get_shape_id()
    coordinates = Affloader.get_coordinates()
    labels = Affloader.get_label()
    affordances = Affloader.get_affordance()
    #Affloader.generate_pointcloud()
    semantic_labels = Affloader.get_semantic_class()
    Affloader.generate_clip_sentences()
    clip_sentences = Affloader.get_clip_sentences()

    # rnumber = random.randint(0, len(semantic_labels) - 1)
    rnumber = 0
    shape_id = shape_ids[rnumber]
    coordinate = coordinates[rnumber]
    label = labels[rnumber]
    affordance = affordances[rnumber]
    clip_sentence = clip_sentences[rnumber]
    semantic_label = semantic_labels[rnumber]

    pathfile = path + semantic_label + "/" + shape_id + ".ply"
    objfile = path + semantic_label + "/" + shape_id + ".obj"

    for i in range(len(clip_sentence)):
        key = list(label.keys())[i]  # Get the first key
        value = label[key]
        print(clip_sentence[i])
    #   optimize_affonet(args, clip_sentence=clip_sentence[i], filepath=pathfile, gt_label=value, obj=objfile)

    #using voxel mesh
    if args.voxel:
        if data is None:
            args, clip_text, obj_file_path, labels = create_voxel_from_mesh(args)
            args.obj_path = obj_file_path
            for i in range(len(labels)):
                args.classes = labels[i]
                args.prompt = clip_text[i]
                args.output_dir = f'voxel_results/demo_{args.object}_{labels[i]}'
                optimize(args)
        else:
            args, clip_text, obj_file_path, labels = voxel_from_mesh(args, Affloader)
            args.obj_path = obj_file_path
            for i in range(len(labels)):
                args.classes = labels[i]
                args.prompt = clip_text[i]
                args.output_dir = f'voxel_results/demo_{args.object}_{labels[i]}'
                optimize(args)