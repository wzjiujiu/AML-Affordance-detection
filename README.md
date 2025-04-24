# 3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions[AML Project]

[Liu Yonghu], [He Haochen], [Kang Chenghua]

Abstract: *The initial project is based on 3D Highlighter, a technique for
localizing semantic regions on a mesh using text as input. This
approach eliminates the need to directly use 3D data for training
a predictive model. Instead, it leverages rendered images from
different viewpoints of the 3D model, allowing direct highlighting
on the 2D images, especially when 3D data is insufficient.
The goal of this experiment is to adapt the project from 3D
mesh data, which represents surfaces using structured elements
(vertices, edges, and faces), to 3D point cloud data, an unstruc-
tured set of points capturing object geometry. This adaptation
explores the performance of the approach on point cloud data
and analyzes how it differs from its application on mesh data.*

<a href="https://github.com/wzjiujiu/AML-Affordance-detection/tree/main"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https%3A%2F%2Fpals.ttic.edu%2Fp%2Fscore-jacobian-chaining" height=22.5></a>

![teaser](./media/teaser.png)


## Installation

Install and activate the conda environment with the following commands. 
```
conda env create --file 3DHighlighter.yml
conda activate 3DHighlighter
```
Note: The installation will fail if run on something other than a CUDA GPU machine.

#### System Requirements
- Python 3.9
- CUDA 11
- 16 GB GPU

#### Installtion (using conda)
- If pip install 'clip' and 'kaolin' failed
- Install git first and check the right version for kaolin
```
conda install git
```
- Replace TORCH_VERSION and CUDA_VERSION with your torch / cuda versions
```

pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VERSION}_cu{CUDA_VERSION}.html
```
- Using the [link](https://pytorch.org/get-started/previous-versions/) to check the right version
- Using the command in demo of [3DHighlighter.ipynb](3DHighlighter.ipynb)

## 🔧 Key Modules and Functions

### `main.py`

The entry point of the pipeline. This script handles configuration parsing, dataset loading, and orchestrates the training or evaluation workflow. It performs the following steps:
- Parses user-defined arguments (e.g., resolution, learning rate, number of views).
- Loads and preprocesses input data (point cloud or voxel grid).
- Initializes the model and optimizer.
- Calls the `render` function to generate images from 3D views.
- Performs training using CLIP-based loss and logs the results.

### `render.py`
Implements differentiable multi-view rendering logic using neural or geometric techniques. It is used to generate training supervision signals (rendered images) from 3D inputs.

### `mesh.py`
Responsible for converting voxel grids into triangle meshes for visualization and further processing. It also ensures proper mesh orientation by computing normals.

### `approximate_mesh.py`
Uses Open3D's TriangleMesh to construct coarse mesh approximations from 3D data. This method is primarily experimental and is only partially used for visualization.

### `converter.py`
Handles the conversion between point clouds and voxel grids. It includes resolution control, normalization, and binary occupancy tensor generation from raw point coordinates.

### `neural_highligter.py`
The core learning module. Implements the CLIP-guided neural affordance highlighter, integrating rendering, prompt encoding, loss computation, and gradient-based optimization.

### `o3dmesh.py`
Contains utilities for working with Open3D mesh structures, such as mesh cleaning, visual inspection, normal estimation, and geometry export.

### `Point_cloud_render.py`
Implements point cloud rendering using projection and camera simulation. It converts 3D point clouds into 2D RGB images for CLIP-based supervision.

### `voxel_mesh.py`
Encapsulates the voxel-based affordance prediction workflow. Handles voxelization, mesh extraction, training, evaluation, and result visualization using voxel grids.

### `utils.py`
A collection of utility functions for file handling, projection matrix computation, view sampling, logging, and visualizations like affordance heatmaps.

### `try.py`
A testbed script used for rapid prototyping and debugging. Typically runs isolated functions or experimental code snippets to validate core components.

### Examples prompt that we used
````
"pull": "a handle for pulling",
"listen": "a part for listening",
"wear": "a wearable section",
"press": "a button for pressing",
````
```
"pull": "a highlighted handle or grip designed for pulling motion",
"listen": "a highlighted speaker grille or microphone for audio input or output",
"wear": "a highlighted strap, loop, or adjustable section for wearing",
"press": "a highlighted depressible button with clear activation feedback",
```
### Note on Reproducibility
Due to small non-determinisms in CLIP's backward process and the sensitivity of our optimization, results can vary across different runs even when fully seeded. If the result of the optimization does not match the expected result, try re-running the optimization.

## Tips for Troubleshooting New Mesh+Region Combinations:
- Due to the sensitivity of the optimization process, if a mesh+prompt combination does not work on the first try, rerun the optimization with a new seed as it might just have found a bad local minimum.
- If an initial specification of a region does not work well, try describing that region with more specific language (i.e. 'eyeglasses' instead of 'glasses'). Also, try using a different target localization text that might correspond to a similar region (i.e. using 'headphones' or 'earmuffs' instead of 'ears').
- In our experiments, we found that using gray and highlighter colors and the prompt format of `"A 3D render of a gray [object] with highlighted [region]"` works best for most mesh+region combinations. However, we encourage users to edit the code to try different prompt specifications since different wordings might work better with new and different mesh+region combinations.
- The traingulation of the mesh is important. Meshes containing long, skinny triangles and/or small numbers of vertices can lead to bad optimizations.
