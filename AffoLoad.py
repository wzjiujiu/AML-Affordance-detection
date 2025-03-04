import pickle
import numpy as np
import os
import open3d as o3d



class PklLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.shape_id=[]
        self.semantic_class=[]
        self.affordance=[]
        self.label = []  # List of dictionaries for labels
        self.coordinates=[]
        self.affordance_descriptions = {
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
        self.clip_sentences=[]




    def load_data(self):
        try:
            with open(self.file_path, 'rb') as file:
                self.data = pickle.load(file)
                if isinstance(self.data, list) and all(isinstance(item, dict) for item in self.data):
                    print(f"Loaded data successfully. It contains {len(self.data)} dictionaries.")

                    for index, info in enumerate(self.data):
                        self.shape_id.append(info['shape_id'])
                        self.semantic_class.append(info['semantic class'])

                        full_shape = info['full_shape']
                        label = full_shape['label']
                        coordinates = full_shape['coordinate']
                        self.coordinates.append(coordinates)
                        # Filter out zero arrays in the 'label' dictionary
                        dict_label = self.filter_non_zero_entries(label)
                        self.label.append(dict_label)
                        self.affordance.append(list(dict_label.keys()))



                else:
                    raise ValueError("The file does not contain a list of dictionaries.")
        except Exception as e:
            print(f"Error loading the file: {e}")

    def get_data(self):
        return self.data

    def get_shape_id(self):
        return self.shape_id

    def get_semantic_class(self):
        return self.semantic_class

    def get_affordance(self):
        return self.affordance

    def get_label(self):
        return self.label

    def get_coordinates(self):
        return self.coordinates

    def filter_non_zero_entries(self,input_dict):
        # Create a new dictionary to store the filtered entries
        filtered_dict = {}

        # Iterate over each item in the input dictionary
        for key, value in input_dict.items():
            # Check if the numpy array is not all zeros
            if not np.all(value == 0):
                filtered_dict[key] = value

        return filtered_dict

    def generate_pointcloud(self):

        output_dir = "point_clouds"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for item in self.data:
            # Extract 3D coordinates
            coordinates = item['full_shape']['coordinate']

            # Create the point cloud object
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(coordinates)

            # Get semantic label (e.g., 'Door')
            label = item['semantic class']

            # Create directory for each label
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            # Generate file paths
            shape_id = item['shape_id']
            ply_file = os.path.join(label_dir, f"{shape_id}.ply")
            obj_file = os.path.join(label_dir, f"{shape_id}.obj")

            # Save PLY point cloud
            o3d.io.write_point_cloud(ply_file, point_cloud, write_ascii=True)
            print(f"Saved point cloud for shape {shape_id} as {ply_file}")

            # Convert point cloud to mesh (Alpha Shape reconstruction)
            alpha = 0.03  # Adjust alpha parameter as needed
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
                mesh.compute_vertex_normals()

                # Save mesh as OBJ
                o3d.io.write_triangle_mesh(obj_file, mesh)
                print(f"Saved mesh for shape {shape_id} as {obj_file}")

            except Exception as e:
                print(f"Failed to generate OBJ mesh for shape {shape_id}: {e}")

        print("Point cloud and mesh generation completed.")

    def generate_clip_sentences(self):
        for i in range(len(self.semantic_class)):
            semantic_class = self.semantic_class[i]
            affordances = self.affordance[i]

            # Generate a list of descriptions for each affordance
            descriptions = [
                f"A 3D render of {semantic_class.lower()} with {self.affordance_descriptions.get(a, f'a highlighted {a}')}"
                for a in affordances
            ]

            # Append the list of descriptions for this item
            self.clip_sentences.append(descriptions)

    def get_clip_sentences(self):
        return self.clip_sentences

