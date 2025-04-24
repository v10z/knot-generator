import numpy as np
import pyvista as pv
from scipy.interpolate import CubicHermiteSpline
import argparse

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

def generate_figure_eight_segment(num_points=100, t_start=0, t_end=2*np.pi):
    """Figure-Eight Knot (4_1)"""
    t = np.linspace(t_start, t_end, num_points)
    x = (2 + np.cos(2 * t)) * np.cos(3 * t)
    y = (2 + np.cos(2 * t)) * np.sin(3 * t)
    z = np.sin(4 * t)
    points = np.vstack((x, y, z)).T
    return points

def generate_cinquefoil_segment(num_points=100, t_start=0, t_end=2*np.pi):
    """Cinquefoil Knot (5_1, also (5,2) Torus Knot)"""
    t = np.linspace(t_start, t_end, num_points)
    x = np.cos(2 * t) * (3 + np.cos(5 * t))
    y = np.sin(2 * t) * (3 + np.cos(5 * t))
    z = np.sin(5 * t)
    points = np.vstack((x, y, z)).T * 0.5 # Scale down slightly
    return points

def generate_torus_knot_segment(p, q, num_points=100, t_start=0, t_end=2*np.pi, R=3, r=1):
    """Generic Torus Knot (p,q)"""
    t = np.linspace(t_start, t_end, num_points)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = -r * np.sin(q * t)
    points = np.vstack((x, y, z)).T * 0.5 # Scale down slightly
    return points

def generate_stevedore_segment(num_points=100, t_start=0, t_end=2*np.pi):
    """Stevedore Knot (6_1)"""
    t = np.linspace(t_start, t_end, num_points)
    x = np.cos(t) + 2*np.cos(3*t)
    y = np.sin(t) - 2*np.sin(3*t)
    z = 2*np.sin(4*t)
    points = np.vstack((x, y, z)).T * 0.5 # Scale down slightly
    return points

def generate_three_twist_segment(num_points=100, t_start=0, t_end=2*np.pi):
    """Three-Twist Knot (5_2)"""
    t = np.linspace(t_start, t_end, num_points)
    x = np.cos(t) * (3 + np.cos(3*t))
    y = np.sin(t) * (3 + np.cos(3*t))
    z = np.sin(4*t) # Approximation, actual 5_2 is more complex
    points = np.vstack((x, y, z)).T * 0.5 # Scale down slightly
    return points

def generate_lissajous_segment(num_points=100, t_start=0, t_end=2*np.pi, a=3, b=4, c=5, delta=np.pi/2):
    """Lissajous Knot"""
    t = np.linspace(t_start, t_end, num_points)
    x = np.cos(a * t)
    y = np.sin(b * t)
    z = np.cos(c * t + delta)
    points = np.vstack((x, y, z)).T * 2.0 # Scale up slightly
    return points

# --- Knot Dictionary ---
knot_generators = {
    "trefoil": generate_trefoil_segment,
    "figure_eight": generate_figure_eight_segment,
    "cinquefoil": generate_cinquefoil_segment,
    "torus_3_2": lambda n, s, e: generate_torus_knot_segment(3, 2, n, s, e),
    "torus_4_3": lambda n, s, e: generate_torus_knot_segment(4, 3, n, s, e),
    "torus_5_2": generate_cinquefoil_segment, # Same as cinquefoil
    "torus_5_3": lambda n, s, e: generate_torus_knot_segment(5, 3, n, s, e),
    "stevedore": generate_stevedore_segment,
    "three_twist": generate_three_twist_segment,
    "lissajous_3_4_5": lambda n, s, e: generate_lissajous_segment(n, s, e, a=3, b=4, c=5),
}

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate a continuous 3D knot pattern.")
parser.add_argument(
    "--knot_type",
    type=str,
    default="trefoil",
    choices=list(knot_generators.keys()),
    help="Type of knot to use for the segments."
)
args = parser.parse_args()

# --- 2. Define Repetition Parameters ---

# Number of repetitions in each direction
nx, ny, nz = 3, 3, 2

# Spacing between the start points of repeated motifs
spacing_x = 7.0
spacing_y = 7.0
spacing_z = 4.0

# Number of points for discretizing each knot segment curve
num_points_per_knot = 100

# Get the selected knot generation function
selected_knot_func = knot_generators[args.knot_type]

# Generate the points for the base knot motif using the selected function
base_knot_points = selected_knot_func(num_points_per_knot, 0, 2*np.pi)
print(f"Using knot type: {args.knot_type}")

# --- 3. Generate the Repeating Pattern ---

# Calculate grid center offset
center_offset_x = (nx - 1) * spacing_x / 2.0
center_offset_y = (ny - 1) * spacing_y / 2.0
center_offset_z = (nz - 1) * spacing_z / 2.0
grid_center_offset = np.array([center_offset_x, center_offset_y, center_offset_z])

layer_points_lists = {k: [] for k in range(nz)} # Store points per layer
layer_knot_start_points = {k: {} for k in range(nz)} # Store start points per layer {(i,j): point}
num_interp_points = 60 # Number of points for spline interpolation between segments

# --- Generate points with serpentine path within each layer ---
print("Generating points for each layer...")
for k in range(nz):
    last_segment_points_in_layer = None
    # Iterate through rows (j)
    for j in range(ny):
        # Determine direction for this row (i)
        if j % 2 == 0:
            i_range = range(nx)
        else:
            i_range = range(nx - 1, -1, -1)

        # Iterate through columns (i) in the determined direction
        for i in i_range:
            # Calculate the raw offset for this instance
            raw_offset = np.array([i * spacing_x, j * spacing_y, k * spacing_z])
            # Apply centering offset
            centered_offset = raw_offset - grid_center_offset

            # Translate the base knot points using the centered offset
            current_segment_points = base_knot_points + centered_offset
            layer_knot_start_points[k][(i, j)] = current_segment_points[0] # Store start point

            # Add smooth interpolation points if not the first segment *in this layer*
            if last_segment_points_in_layer is not None:
                p0 = last_segment_points_in_layer[-1]
                m0 = p0 - last_segment_points_in_layer[-2]
                p1 = current_segment_points[0]
                m1 = current_segment_points[1] - p1
                distance = np.linalg.norm(p1 - p0)
                if distance > 1e-6:
                    m0_scaled = m0 * distance
                    m1_scaled = m1 * distance
                else:
                    m0_scaled = m0
                    m1_scaled = m1
                times = [0, 1]
                points = np.array([p0, p1])
                tangents = np.array([m0_scaled, m1_scaled])
                spline = CubicHermiteSpline(times, points, tangents)
                interp_times = np.linspace(0, 1, num_interp_points + 2)[1:-1]
                interpolation_points = spline(interp_times)
                if interpolation_points.size > 0:
                    layer_points_lists[k].append(interpolation_points)

            # Add the current segment points to the correct layer list
            layer_points_lists[k].append(current_segment_points)

            # Update the last segment points for the next iteration *in this layer*
            last_segment_points_in_layer = current_segment_points

# Combine points for each layer
layer_combined_points = {}
for k in range(nz):
    if not layer_points_lists[k]:
        print(f"Warning: No points generated for layer {k}.")
        layer_combined_points[k] = np.empty((0, 3))
    else:
        layer_combined_points[k] = np.concatenate(layer_points_lists[k], axis=0)

# --- 4. Visualize using PyVista ---

# Create a PyVista plotter object
plotter = pv.Plotter(window_size=[800, 800])
plotter.set_background('white') # Set background to white

knot_tube_radius = 0.2
connection_tube_radius = 0.1
num_conn_interp = 30 # Number of points for inter-layer connection splines

print("Adding layer paths to plotter...")
# Add each layer's path
layer_colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'] # Add more if nz > 4
for k in range(nz):
    points_array = layer_combined_points[k]
    if points_array.shape[0] > 1:
        poly = pv.PolyData()
        poly.points = points_array
        num_total_points = points_array.shape[0]
        lines = np.full((num_total_points - 1, 3), 2, dtype=np.int_)
        lines[:, 1] = np.arange(0, num_total_points - 1, dtype=np.int_)
        lines[:, 2] = np.arange(1, num_total_points, dtype=np.int_)
        poly.lines = lines
        tube = poly.tube(radius=knot_tube_radius)
        plotter.add_mesh(tube, color=layer_colors[k % len(layer_colors)], smooth_shading=True)
    else:
        print(f"Not enough points to draw lines for layer {k}.")

# Add 4 corner connections between layer 0 and layer 1 (if nz >= 2)
print("Adding inter-layer connections...")
if nz >= 2:
    corner_indices = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    for i, j in corner_indices:
        if (i, j) in layer_knot_start_points[0] and (i, j) in layer_knot_start_points[1]:
            p0 = layer_knot_start_points[0][(i, j)] # Start point in layer 0
            p1 = layer_knot_start_points[1][(i, j)] # Start point in layer 1

            # Use direction vector as tangent for both ends
            tangent = p1 - p0
            times = [0, 1]
            points = np.array([p0, p1])
            tangents = np.array([tangent, tangent]) # Same tangent for start/end
            spline = CubicHermiteSpline(times, points, tangents)
            interp_times = np.linspace(0, 1, num_conn_interp)
            spline_points = spline(interp_times)
            spline_poly = pv.lines_from_points(spline_points)
            conn_tube = spline_poly.tube(radius=connection_tube_radius)
            plotter.add_mesh(conn_tube, color='gray', smooth_shading=True)
        else:
             print(f"Warning: Corner knot ({i},{j}) missing in layer 0 or 1, cannot connect.")


# Add axes for reference
plotter.add_axes()

# Show the plot (opens an interactive window)
print(f"Displaying the layered {args.knot_type} pattern with corner connections. Close the window to exit.")
plotter.show()

print("Done.")
