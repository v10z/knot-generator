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

all_points_list = [] # List to store points sequentially
last_segment_points = None
num_interp_points = 60 # Increased number of points for spline interpolation between segments

# --- Generate points with serpentine path ---
# Iterate through layers (k)
for k in range(nz):
    # Iterate through rows (j)
    for j in range(ny):
        # Determine direction for this row (i)
        # Even rows go forwards (0 to nx-1), odd rows go backwards (nx-1 to 0)
        if j % 2 == 0:
            i_range = range(nx)
        else:
            i_range = range(nx - 1, -1, -1)

        # Iterate through columns (i) in the determined direction
        for i in i_range:
            # Calculate the offset for this instance
            offset = np.array([i * spacing_x, j * spacing_y, k * spacing_z])

            # Translate the base knot points (using the pre-generated trefoil)
            current_segment_points = base_knot_points + offset

            # Add smooth interpolation points if not the first segment
            if last_segment_points is not None:
                # End point of the last segment
                p0 = last_segment_points[-1]
                # Approximate tangent at the end of the last segment (vector pointing away from the end)
                m0 = p0 - last_segment_points[-2]
                # Start point of the current segment
                p1 = current_segment_points[0]
                # Approximate tangent at the start of the current segment (vector pointing away from the start)
                m1 = current_segment_points[1] - p1

                # Scale tangents by the distance between points to improve spline shape
                distance = np.linalg.norm(p1 - p0)
                # Avoid division by zero or tiny distances causing huge tangents
                if distance > 1e-6:
                    m0_scaled = m0 * distance
                    m1_scaled = m1 * distance
                else:
                    m0_scaled = m0
                    m1_scaled = m1

                # Create the spline
                # We define the spline at t=0 and t=1
                times = [0, 1]
                points = np.array([p0, p1])
                tangents = np.array([m0_scaled, m1_scaled]) # Use scaled tangents

                spline = CubicHermiteSpline(times, points, tangents)

                # Evaluate the spline at intermediate points (excluding endpoints)
                interp_times = np.linspace(0, 1, num_interp_points + 2)[1:-1]
                interpolation_points = spline(interp_times)

                if interpolation_points.size > 0: # Only add if num_interp_points > 0
                    all_points_list.append(interpolation_points)

            # Add the current segment points
            all_points_list.append(current_segment_points)

            # Update the last segment points for the next iteration's interpolation
            last_segment_points = current_segment_points

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
