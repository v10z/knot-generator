import numpy as np
import pyvista as pv
from scipy.interpolate import CubicHermiteSpline
import argparse
import os

from .knots import knot_generators

def generate_continuous_knot(knot_type, nx, ny, nz, spacing_x, spacing_y, spacing_z, num_points_per_knot, num_interp_points):
    """
    Generates points for a continuous 3D knot pattern.

    Args:
        knot_type (str): Type of knot to use for the segments.
        nx (int): Number of repetitions in the x direction.
        ny (int): Number of repetitions in the y direction.
        nz (int): Number of repetitions in the z direction.
        spacing_x (float): Spacing between the start points of repeated motifs in x.
        spacing_y (float): Spacing between the start points of repeated motifs in y.
        spacing_z (float): Spacing between the start points of repeated motifs in z.
        num_points_per_knot (int): Number of points for discretizing each knot segment curve.
        num_interp_points (int): Number of points for spline interpolation between segments.

    Returns:
        np.ndarray: An array of shape (N, 3) containing XYZ coordinates for the continuous knot.
    """
    if knot_type not in knot_generators:
        raise ValueError(f"Unknown knot type: {knot_type}. Available types: {list(knot_generators.keys())}")

    selected_knot_func = knot_generators[knot_type]
    base_knot_points = selected_knot_func(num_points_per_knot, 0, 2*np.pi)

    # Calculate grid center offset
    center_offset_x = (nx - 1) * spacing_x / 2.0
    center_offset_y = (ny - 1) * spacing_y / 2.0
    center_offset_z = (nz - 1) * spacing_z / 2.0
    grid_center_offset = np.array([center_offset_x, center_offset_y, center_offset_z])

    all_points_list = [] # List to store points sequentially
    last_segment_points = None

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
                # Calculate the raw offset for this instance
                raw_offset = np.array([i * spacing_x, j * spacing_y, k * spacing_z])
                # Apply centering offset
                centered_offset = raw_offset - grid_center_offset

                # Translate the base knot points using the centered offset
                current_segment_points = base_knot_points + centered_offset

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
        return np.empty((0, 3))
    else:
        return np.concatenate(all_points_list, axis=0)

def visualize_knot(points_array, knot_type):
    """
    Visualizes the generated knot using PyVista.

    Args:
        points_array (np.ndarray): Array of XYZ coordinates.
        knot_type (str): Type of knot for display in the title.
    """
    plotter = pv.Plotter(window_size=[800, 800])
    plotter.set_background('white')

    if points_array.shape[0] > 1:
        poly = pv.PolyData()
        poly.points = points_array
        num_total_points = points_array.shape[0]
        lines = np.full((num_total_points - 1, 3), 2, dtype=np.int_)
        lines[:, 1] = np.arange(0, num_total_points - 1, dtype=np.int_)
        lines[:, 2] = np.arange(1, num_total_points, dtype=np.int_)
        poly.lines = lines

        tube = poly.tube(radius=0.2)
        plotter.add_mesh(tube, smooth_shading=True, cmap='viridis', scalar_bar_args={'title': "Path Progression"})
    else:
        print("Not enough points to draw lines.")

    plotter.add_axes()
    plotter.show(title=f"Continuous {knot_type} Pattern")

def save_knot_data(points_array, output_path):
    """
    Saves the generated knot points to a file.

    Args:
        points_array (np.ndarray): Array of XYZ coordinates.
        output_path (str): Path to save the output file.
    """
    try:
        np.savetxt(output_path, points_array)
        print(f"Knot data saved to {output_path}")
    except Exception as e:
        print(f"Error saving knot data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate and visualize a continuous 3D knot pattern.")
    parser.add_argument(
        "--knot_type",
        type=str,
        default="trefoil",
        choices=list(knot_generators.keys()),
        help="Type of knot to use for the segments."
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=3,
        help="Number of repetitions in the x direction."
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=3,
        help="Number of repetitions in the y direction."
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=2,
        help="Number of repetitions in the z direction."
    )
    parser.add_argument(
        "--spacing_x",
        type=float,
        default=7.0,
        help="Spacing between the start points of repeated motifs in x."
    )
    parser.add_argument(
        "--spacing_y",
        type=float,
        default=7.0,
        help="Spacing between the start points of repeated motifs in y."
    )
    parser.add_argument(
        "--spacing_z",
        type=float,
        default=4.0,
        help="Spacing between the start points of repeated motifs in z."
    )
    parser.add_argument(
        "--num_points_per_knot",
        type=int,
        default=100,
        help="Number of points for discretizing each knot segment curve."
    )
    parser.add_argument(
        "--num_interp_points",
        type=int,
        default=60,
        help="Number of points for spline interpolation between segments."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output file path to save the generated points (e.g., knot_points.txt)."
    )

    args = parser.parse_args()

    print(f"Generating continuous {args.knot_type} pattern...")

    knot_points = generate_continuous_knot(
        args.knot_type,
        args.nx,
        args.ny,
        args.nz,
        args.spacing_x,
        args.spacing_y,
        args.spacing_z,
        args.num_points_per_knot,
        args.num_interp_points
    )

    if args.output:
        save_knot_data(knot_points, args.output)

    if knot_points.shape[0] > 1:
        print(f"Displaying the continuous {args.knot_type} pattern. Close the window to exit.")
        visualize_knot(knot_points, args.knot_type)
    else:
        print("No points generated to visualize.")

    print("Done.")

if __name__ == "__main__":
    main()
