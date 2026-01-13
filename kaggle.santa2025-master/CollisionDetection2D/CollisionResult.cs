using kaggle.santa2025.Geometry2D;

namespace kaggle.santa2025.CollisionDetection2D
{
    /// <summary>
    /// Result of a collision query, including penetration depth and normal.
    /// </summary>
    public struct CollisionResult
    {
        public bool IsColliding;           // True if shapes intersect
        public Vector2D PenetrationNormal; // Normal pointing from shape B to shape A (direction to push A out of B)
        public double PenetrationDepth;    // Magnitude of the minimum translation vector
    }
}
