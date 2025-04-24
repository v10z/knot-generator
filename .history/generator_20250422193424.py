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

all_points_list = [] # List to store points sequentially
last_point = None
num_interp_points = 10 # Number of points for interpolation between segments

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            # Calculate the offset for this instance
            offset = np.array([i * spacing_x, j * spacing_y, k * spacing_z])

            # Translate the base knot points
            current_segment_points = base_knot_points + offset

            # Add interpolation points if not the first segment
            if last_point is not None:
                # Generate points for a straight line connection (excluding endpoints)
                interp_x = np.linspace(last_point[0], current_segment_points[0, 0], num_interp_points + 2)[1:-1]
                interp_y = np.linspace(last_point[1], current_segment_points[0, 1], num_interp_points + 2)[1:-1]
                interp_z = np.linspace(last_point[2], current_segment_points[0, 2], num_interp_points + 2)[1:-1]
                interpolation_points = np.vstack((interp_x, interp_y, interp_z)).T
                if interpolation_points.size > 0: # Only add if num_interp_points > 0
                    all_points_list.append(interpolation_points)

            # Add the current segment points
            all_points_list.append(current_segment_points)

            # Update the last point for the next iteration's interpolation
            last_point = current_segment_points[-1]

# Combine all points into a single array
if not all_points_list:
    print("Warning: No points generated.")
    all_points_array = np.empty((0, 3))
else:
    all_points_array = np.concatenate(all_points_list, axis=0)


# --- 4. Visualize using PyVista ---

# Create a PyVista plotter object
plotter = pv.Plotter(window_size=[800, 800])
plotter.set_background('white') # Set background to white

# Check if there are points to plot
if all_points_array.shape[0] > 1:
    # Create a PyVista PolyData object representing the single continuous line/curve
    poly = pv.PolyData()
    poly.points = all_points_array
    # Create line connectivity ( N points need N-1 line segments: 0-1, 1-2, ...)
    num_total_points = all_points_array.shape[0]
    lines = np.full((num_total_points - 1, 3), 2, dtype=np.int_)
    lines[:, 1] = np.arange(0, num_total_points - 1, dtype=np.int_)
    lines[:, 2] = np.arange(1, num_total_points, dtype=np.int_)
    poly.lines = lines

    # Create a tube around the line for better visualization
    tube = poly.tube(radius=0.2)

    # Add the tube mesh to the plotter
    # Use a colormap along the path to show progression
    plotter.add_mesh(tube, smooth_shading=True, cmap='viridis', scalar_bar_args={'title': "Path Progression"})
    # Or add the simple line:
    # plotter.add_mesh(poly, color='blue', line_width=3)
else:
    print("Not enough points to draw lines.")


# Add axes for reference
plotter.add_axes()

# Show the plot (opens an interactive window)
print("Displaying the continuous 3D pattern. Close the window to exit.")
plotter.show()

print("Done.")
