import numpy as np

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
