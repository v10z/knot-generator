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

# --- 3. Generate the Knot Grid Data ---

knot_data = {} # Dictionary to store knot points and start point: {(i, j, k): {'points': array, 'start_point': array}}

print("Generating knot grid...")
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            # Calculate the offset for this instance
            offset = np.array([i * spacing_x, j * spacing_y, k * spacing_z])
            # Translate the base knot points
            current_segment_points = base_knot_points + offset
            # Store the points and the first point (for connections)
            knot_data[(i, j, k)] = {
                'points': current_segment_points,
                'start_point': current_segment_points[0] # Use first point as anchor
            }

# --- 4. Visualize using PyVista ---

# Create a PyVista plotter object
plotter = pv.Plotter(window_size=[800, 800])
plotter.set_background('white') # Set background to white

connection_tube_radius = 0.1
knot_tube_radius = 0.2

print("Adding knots and connections to plotter...")
# Add Knots
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            data = knot_data[(i, j, k)]
            points = data['points']

            if points.shape[0] > 1:
                poly = pv.PolyData()
                poly.points = points
                num_knot_pts = points.shape[0]
                lines = np.full((num_knot_pts - 1, 3), 2, dtype=np.int_)
                lines[:, 1] = np.arange(0, num_knot_pts - 1, dtype=np.int_)
                lines[:, 2] = np.arange(1, num_knot_pts, dtype=np.int_)
                poly.lines = lines
                knot_tube = poly.tube(radius=knot_tube_radius)
                plotter.add_mesh(knot_tube, color='lightblue', smooth_shading=True)

# Add Connections (Straight tubes between start points of adjacent knots)
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            start_node = knot_data[(i, j, k)]['start_point']

            # Connect to neighbors with smooth splines
            num_conn_interp = 30 # Number of points for connection splines

            # Connect to neighbor in +i direction
            if i + 1 < nx:
                p0 = start_node
                p1 = knot_data[(i + 1, j, k)]['start_point']
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

            # Connect to neighbor in +j direction
            if j + 1 < ny:
                p0 = start_node
                p1 = knot_data[(i, j + 1, k)]['start_point']
                tangent = p1 - p0
                times = [0, 1]
                points = np.array([p0, p1])
                tangents = np.array([tangent, tangent])
                spline = CubicHermiteSpline(times, points, tangents)
                interp_times = np.linspace(0, 1, num_conn_interp)
                spline_points = spline(interp_times)
                spline_poly = pv.lines_from_points(spline_points)
                conn_tube = spline_poly.tube(radius=connection_tube_radius)
                plotter.add_mesh(conn_tube, color='gray', smooth_shading=True)

            # Connect to neighbor in +k direction
            if k + 1 < nz:
                p0 = start_node
                p1 = knot_data[(i, j, k + 1)]['start_point']
                tangent = p1 - p0
                times = [0, 1]
                points = np.array([p0, p1])
                tangents = np.array([tangent, tangent])
                spline = CubicHermiteSpline(times, points, tangents)
                interp_times = np.linspace(0, 1, num_conn_interp)
                spline_points = spline(interp_times)
                spline_poly = pv.lines_from_points(spline_points)
                conn_tube = spline_poly.tube(radius=connection_tube_radius)
                plotter.add_mesh(conn_tube, color='gray', smooth_shading=True)


# Add axes for reference
plotter.add_axes()

# Show the plot (opens an interactive window)
print(f"Displaying the {args.knot_type} lattice structure. Close the window to exit.")
plotter.show()

print("Done.")
