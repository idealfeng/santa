"""Static polygon data for the Santa 2025 tree shape.

`TREE_POINTS` defines the official 15-vertex (x, y) polygon used in the
competition baselines and in this repository. The polygon is intended to be
used in local coordinates (centered around the origin) and then transformed
with translation/rotation per tree.
"""

TREE_POINTS = [
    [0.0, 0.8],  # Tip
    [0.125, 0.5],  # Top tier (outer)
    [0.0625, 0.5],  # Top tier (inner)
    [0.2, 0.25],  # Mid tier (outer)
    [0.1, 0.25],  # Mid tier (inner)
    [0.35, 0.0],  # Bottom tier (outer)
    [0.075, 0.0],  # Trunk (top-right)
    [0.075, -0.2],  # Trunk (bottom-right)
    [-0.075, -0.2],  # Trunk (bottom-left)
    [-0.075, 0.0],  # Trunk (top-left)
    [-0.35, 0.0],  # Bottom tier (outer)
    [-0.1, 0.25],  # Mid tier (inner)
    [-0.2, 0.25],  # Mid tier (outer)
    [-0.0625, 0.5],  # Top tier (inner)
    [-0.125, 0.5],  # Top tier (outer)
]
