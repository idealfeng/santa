using kaggle.santa2025.Geometry2D;

namespace kaggle.santa2025.CollisionDetection2D
{
    /// <summary>
    /// Represents an edge in the expanding polytope for EPA.
    /// </summary>
    public struct PolytopeEdge
    {
        public int IndexA;               // Index of first vertex in the polytope list
        public int IndexB;               // Index of second vertex in the polytope list
        public Vector2D Normal;          // Outward-pointing normal (perpendicular to edge, pointing away from origin)
        public double Distance;          // Distance from origin to the edge (positive if origin outside)
    }
}
