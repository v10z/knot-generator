import numpy as np
import pyvista as pv

# --- 1. Define the Base Knot Motif ---

def generate_trefoil_segment(num_points=100, t_start=0, t_end=2*np.pi):
    """
    Generates points for a segment of a trefoil knot.

    Args:
        num_points (int): Number of points to generate along the segment.
        t_start (float): Starting parameter value.
        t_end (float): Ending parameter value (default 2*pi for a full loop).

    Returns:
        np.ndarray: An array of shape (num_points, 3) containing XYZ coordinates.
    """
    t = np.linspace(t_start, t_end, num_points)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)

    # Stack coordinates into an (N, 3) array
    points = np.vstack((x, y, z)).T
    return points

# --- 2. Define Repetition Parameters ---

# Number of repetitions in each direction
nx, ny, nz = 3, 3, 2

# Spacing between the start points of repeated motifs
spacing_x = 7.0
spacing_y = 7.0
spacing_z = 4.0

# Number of points for discretizing each knot segment curve
num_points_per_knot = 100

# Generate the points for the base knot motif (one full loop here for simplicity)
base_knot_points = generate_trefoil_segment(num_points_per_knot, 0, 2*np.pi)

# --- 3. Generate the Repeating Pattern ---

all_knot_segments = [] # List to store the points of each translated knot

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            # Calculate the offset for this instance
            offset = np.array([i * spacing_x, j * spacing_y, k * spacing_z])

            # Translate the base knot points
            translated_knot_points = base_knot_points + offset

            # Store the points for this segment
            all_knot_segments.append(translated_knot_points)

# --- 4. Visualize using PyVista ---

# Create a PyVista plotter object
plotter = pv.Plotter(window_size=[800, 800])
plotter.set_background('white') # Set background to white

# Add each knot segment to the plotter
# Using a colormap to distinguish the segments
cmap = pv.LookupTable('viridis', n_values=len(all_knot_segments))

for idx, points in enumerate(all_knot_segments):
    # Create a PyVista PolyData object representing the line/curve
    poly = pv.PolyData()
    poly.points = points
    # Create line connectivity ( N points need N-1 line segments: 0-1, 1-2, ...)
    lines = np.full((len(points)-1, 3), 2, dtype=np.int_)
    lines[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    lines[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = lines

    # Create a tube around the line for better visualization
    tube = poly.tube(radius=0.2)

    # Add the tube mesh to the plotter with a specific color
    plotter.add_mesh(tube, color=cmap.map_value(idx), smooth_shading=True)
    # Or add the simple line:
    # plotter.add_mesh(poly, color=cmap.map_value(idx), line_width=3)


# Add axes for reference
plotter.add_axes()

# Show the plot (opens an interactive window)
print("Displaying the 3D pattern. Close the window to exit.")
plotter.show()

print("Done.")