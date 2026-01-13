using kaggle.santa2025.Geometry2D;

namespace kaggle.santa2025.CollisionDetection2D
{
    /// <summary>
    /// Represents a point in the Minkowski difference, storing the support points from each shape.
    /// This allows reconstruction of contact points on original shapes if needed.
    /// </summary>
    public struct MinkowskiVertex
    {
        public Vector2D Position;        // Position in Minkowski space: supportA - supportB
        public Vector2D SupportPointA;   // Support point on shape A
        public Vector2D SupportPointB;   // Support point on shape B
    }
}
