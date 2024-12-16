import pyvista as pv


def highlight_mesh_portion(mesh_file, highlight_cells_range):
    """
    Visualize a 3D mesh and highlight a specific portion.

    Parameters:
        mesh_file (str): Path to the mesh file (e.g., OBJ file).
        highlight_cells_range (range): Range of cell indices to highlight.

    Returns:
        None
    """
    # Load the mesh file
    mesh = pv.read(mesh_file)

    # Extract the portion of the mesh to highlight
    highlighted_portion = mesh.extract_cells(highlight_cells_range)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the full mesh to the plotter
    plotter.add_mesh(mesh, color="white", show_edges=True)

    # Add the highlighted portion to the plotter
    plotter.add_mesh(highlighted_portion, color="red", show_edges=True, line_width=2)

    # Add a light source for better visualization
    light = pv.Light(position=(10, 10, 10), intensity=0.7)
    plotter.add_light(light)

    # Set the view properties
    plotter.show_bounds(grid="front", location="outer", all_edges=True)
    plotter.view_isometric()

    # Render and display the plot
    plotter.show()