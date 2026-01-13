using kaggle.santa2025.Geometry2D;
using System;
using System.Collections.Generic;
using System.Linq;

namespace kaggle.santa2025.CollisionDetection2D
{
    /// <summary>
    /// Main class implementing the 2D GJK algorithm for collision detection
    /// and the EPA algorithm for penetration depth and normal computation.
    /// Works only for convex polygons (vertices must be in counterclockwise order).
    /// </summary>
    public static class GjkEpaCollisionDetector
    {
        private const double Epsilon = 1e-6;                // Tolerance for floating-point comparisons
        private const int MaxIterations = 50;               // Safety limit to prevent infinite loops

        /// <summary>
        /// Computes the support point on a convex polygon in the given direction.
        /// The support point is the vertex farthest in the direction vector.
        /// </summary>
        /// <param name="vertices">List of vertices of the convex polygon (counterclockwise order)</param>
        /// <param name="direction">Direction vector (should be normalized for best results, but not required)</param>
        /// <returns>The vertex farthest in the given direction</returns>
        private static Vector2D ComputeSupportPoint(IReadOnlyList<Vector2D> vertices, Vector2D direction)
        {
            double maximumDotProduct = double.MinValue;
            Vector2D supportPoint = vertices[0];

            for (int i = 0; i < vertices.Count; i++)
            {
                double currentDotProduct = Vector2D.Dot(vertices[i], direction);
                if (currentDotProduct > maximumDotProduct)
                {
                    maximumDotProduct = currentDotProduct;
                    supportPoint = vertices[i];
                }
            }

            return supportPoint;
        }

        /// <summary>
        /// Computes a Minkowski difference vertex: supportA - supportB.
        /// </summary>
        private static MinkowskiVertex ComputeMinkowskiSupport(
            IReadOnlyList<Vector2D> verticesA,
            IReadOnlyList<Vector2D> verticesB,
            Vector2D direction)
        {
            Vector2D supportPointA = ComputeSupportPoint(verticesA, direction);
            Vector2D supportPointB = ComputeSupportPoint(verticesB, -direction); // Negative direction for shape B

            return new MinkowskiVertex
            {
                Position = supportPointA - supportPointB,
                SupportPointA = supportPointA,
                SupportPointB = supportPointB
            };
        }

        /// <summary>
        /// Checks if the origin is inside the line segment AB (including endpoints).
        /// If true, origin is enclosed → collision detected.
        /// </summary>
        private static bool ContainsOriginInLine(ref MinkowskiVertex pointA, ref MinkowskiVertex pointB, out Vector2D newDirection)
        {
            Vector2D vectorAB = pointB.Position - pointA.Position;
            Vector2D vectorAO = -pointA.Position; // From A to origin (origin - A)

            // Perpendicular vector to AB pointing towards origin
            Vector2D perpendicular = new (-vectorAB.Y, vectorAB.X);
            if (Vector2D.Dot(perpendicular, vectorAO) < 0)
                perpendicular = -perpendicular;

            newDirection = perpendicular;
            return false; // Origin cannot be enclosed by a line alone in 2D
        }

        /// <summary>
        /// Checks if the origin is inside the triangle ABC.
        /// Uses triple cross product to determine if origin is on the correct side of each edge.
        /// </summary>
        private static bool ContainsOriginInTriangle(
            ref MinkowskiVertex pointA,
            ref MinkowskiVertex pointB,
            ref MinkowskiVertex pointC,
            out Vector2D newDirection)
        {
            Vector2D vectorAB = pointB.Position - pointA.Position;
            Vector2D vectorAC = pointC.Position - pointA.Position;
            Vector2D vectorAO = -pointA.Position;

            Vector2D perpendicularAB = new (-vectorAB.Y, vectorAB.X);
            Vector2D perpendicularAC = new (-vectorAC.Y, vectorAC.X);

            // Check if origin is to the left of AB, BC, and CA (counterclockwise winding)
            if (Vector2D.Dot(perpendicularAB, vectorAC) < 0) // Wrong winding?
            {
                newDirection = Vector2D.Zero;
                return false;
            }

            if (Vector2D.Dot(perpendicularAB, vectorAO) < -Epsilon)
            {
                // Origin is outside edge AB → reduce to line AB
                pointC = pointA; // Discard C
                newDirection = perpendicularAB;
                return false;
            }

            Vector2D vectorBC = pointC.Position - pointB.Position;
            Vector2D perpendicularBC = new (-vectorBC.Y, vectorBC.X);

            if (Vector2D.Dot(perpendicularBC, -pointB.Position) < -Epsilon)
            {
                // Origin outside BC → reduce to line BC
                pointA = pointB;
                pointC = pointB;
                newDirection = perpendicularBC;
                return false;
            }

            if (Vector2D.Dot(perpendicularAC, -pointA.Position) < -Epsilon)
            {
                // Origin outside CA → reduce to line CA
                pointB = pointA;
                newDirection = perpendicularAC;
                return false;
            }

            // Origin is inside the triangle
            newDirection = Vector2D.Zero;
            return true;
        }

        /// <summary>
        /// Core Gilbert-Johnson-Keerthi (GJK) algorithm for collision detection between two convex shapes.
        /// Returns true if the shapes collide, and outputs the final simplex (triangle or degenerate) for EPA.
        /// This version fixes the simplex management bug by using a fixed array and proper do-while logic.
        /// </summary>
        public static bool GilbertJohnsonKeerthiAlgorithm(
            IReadOnlyList<Vector2D> verticesA,
            IReadOnlyList<Vector2D> verticesB,
            out List<MinkowskiVertex> finalSimplex)
        {
            finalSimplex = null;

            // Use a small fixed array for simplex vertices (max 3 in 2D)
            MinkowskiVertex[] simplex = new MinkowskiVertex[4]; // Extra slot for temporary new vertex
            int simplexCount = 0;

            // Initial direction: pick first vertex of A as starting point
            Vector2D direction = verticesA[0];
            if (direction.LengthSquared() < Epsilon * Epsilon)
                direction = Vector2D.UnitX; // Fallback if shape A is centered at origin

            // First support point
            simplex[0] = ComputeMinkowskiSupport(verticesA, verticesB, direction);
            simplexCount = 1;

            // Search towards origin
            direction = -simplex[0].Position;

            int iterationCount = 0;
            while (iterationCount++ < MaxIterations)
            {
                // Terminate if direction is too small (degenerate case)
                if (direction.LengthSquared() < Epsilon * Epsilon)
                {
                    finalSimplex = [.. simplex.Take(simplexCount)];
                    return true; // Origin is on a vertex → collision
                }

                // Get new support point in current search direction
                MinkowskiVertex newVertex = ComputeMinkowskiSupport(verticesA, verticesB, direction);

                // If new point doesn't pass origin in this direction → origin not in Minkowski difference
                if (Vector2D.Dot(newVertex.Position, direction) <= 0)
                {
                    finalSimplex = null;
                    return false; // No collision
                }

                // Add new vertex temporarily
                simplex[simplexCount] = newVertex;
                simplexCount++;

                // Now process the simplex to see if origin is enclosed and update direction
                if (DoSimplex(simplex, ref simplexCount, ref direction))
                {
                    // Origin is inside the current simplex → collision detected
                    finalSimplex = [.. simplex.Take(simplexCount)];
                    return true;
                }

                // If not enclosed, simplexCount and direction have been updated to closest feature
                // Continue looping with reduced simplex
            }

            // Max iterations reached (very rare for convex shapes)
            finalSimplex = [.. simplex.Take(simplexCount)];
            return true; // Assume collision to avoid false negative
        }

        /// <summary>
        /// Processes the current simplex and determines if the origin is enclosed.
        /// If not, reduces the simplex to the closest feature (line or point) toward the origin
        /// and updates the search direction.
        /// Returns true if origin is enclosed.
        /// </summary>
        private static bool DoSimplex(MinkowskiVertex[] simplex, ref int count, ref Vector2D direction)
        {
            switch (count)
            {
                case 2:
                    return DoLine(simplex, ref count, ref direction);

                case 3:
                    return DoTriangle(simplex, ref count, ref direction);

                case 4:
                    // In 2D, we should never have 4 points — but if we do, fall back to triangle check on last 3
                    count = 3; // Discard oldest
                    return DoTriangle(simplex, ref count, ref direction);

                default:
                    // Should not happen
                    return false;
            }
        }

        /// <summary>
        /// Handles the line segment case (2 points).
        /// Checks if origin is "beyond" the segment and updates direction to perpendicular toward origin.
        /// </summary>
        private static bool DoLine(MinkowskiVertex[] simplex, ref int count, ref Vector2D direction)
        {
            MinkowskiVertex a = simplex[1]; // Most recent
            MinkowskiVertex b = simplex[0]; // Older

            Vector2D ab = b.Position - a.Position;
            Vector2D ao = -a.Position;

            // Perpendicular vector pointing toward origin
            Vector2D perp = new (-ab.Y, ab.X);
            if (Vector2D.Dot(perp, ao) < 0)
                perp = -perp;

            direction = perp;

            // Keep both points (line is closest feature)
            return false;
        }

        /// <summary>
        /// Handles the triangle case (3 points).
        /// Checks if origin is inside triangle. If not, reduces to closest edge.
        /// </summary>
        private static bool DoTriangle(MinkowskiVertex[] simplex, ref int count, ref Vector2D direction)
        {
            MinkowskiVertex a = simplex[2]; // Newest
            MinkowskiVertex b = simplex[1];
            MinkowskiVertex c = simplex[0];

            Vector2D ab = b.Position - a.Position;
            Vector2D ac = c.Position - a.Position;
            Vector2D ao = -a.Position;

            Vector2D abPerp = new (-ab.Y, ab.X);
            Vector2D acPerp = new (-ac.Y, ac.X);

            // Check if origin is outside edge AB
            if (Vector2D.Dot(abPerp, ao) > 0 && Vector2D.Dot(abPerp, ac) < 0 == false)
            {
                // Origin is on the side of AB away from C → closest feature is edge AB
                simplex[0] = simplex[1]; // c := b
                simplex[1] = simplex[2]; // b := a
                count = 2;
                direction = abPerp;
                return false;
            }

            // Check if origin is outside edge AC
            if (Vector2D.Dot(acPerp, ao) > 0)
            {
                // Closest feature is edge AC
                simplex[0] = simplex[2]; // c := a
                count = 2;
                direction = acPerp;
                return false;
            }

            // Origin is inside the triangle (on the correct side of both edges)
            return true;
        }

        /// <summary>
        /// Finds the edge in the polytope closest to the origin.
        /// </summary>
        private static PolytopeEdge FindClosestEdge(List<MinkowskiVertex> polytope)
        {
            PolytopeEdge closestEdge = new PolytopeEdge
            {
                Distance = float.MaxValue
            };

            for (int i = 0; i < polytope.Count; i++)
            {
                int j = (i + 1) % polytope.Count;

                Vector2D edgeVector = polytope[j].Position - polytope[i].Position;
                Vector2D outwardNormal = new (-edgeVector.Y, edgeVector.X); // Rotate 90° CCW
                double normalLengthSquared = outwardNormal.LengthSquared();

                if (normalLengthSquared < Epsilon * Epsilon) continue; // Degenerate edge

                double normalLength = Math.Sqrt(normalLengthSquared);
                outwardNormal /= normalLength;

                // Ensure normal points away from origin (outward)
                if (Vector2D.Dot(outwardNormal, polytope[i].Position) < 0)
                    outwardNormal = -outwardNormal;

                double distance = Vector2D.Dot(outwardNormal, polytope[i].Position);

                if (distance < closestEdge.Distance)
                {
                    closestEdge = new PolytopeEdge
                    {
                        IndexA = i,
                        IndexB = j,
                        Normal = outwardNormal,
                        Distance = distance
                    };
                }
            }

            return closestEdge;
        }

        /// <summary>
        /// Expanding Polytope Algorithm: computes penetration depth and normal using the simplex from GJK.
        /// </summary>
        private static CollisionResult ExpandingPolytopeAlgorithm(
            IReadOnlyList<Vector2D> verticesA,
            IReadOnlyList<Vector2D> verticesB,
            List<MinkowskiVertex> initialSimplex)
        {
            List<MinkowskiVertex> polytope = new List<MinkowskiVertex>(initialSimplex);

            int iterationCount = 0;
            while (iterationCount++ < MaxIterations)
            {
                PolytopeEdge closestEdge = FindClosestEdge(polytope);

                MinkowskiVertex newSupportVertex = ComputeMinkowskiSupport(verticesA, verticesB, closestEdge.Normal);

                double newDistance = Vector2D.Dot(newSupportVertex.Position, closestEdge.Normal);

                // If the new support point does not expand significantly, we have converged
                if (newDistance - closestEdge.Distance < Epsilon)
                {
                    return new CollisionResult
                    {
                        IsColliding = true,
                        PenetrationNormal = closestEdge.Normal,
                        PenetrationDepth = closestEdge.Distance + Epsilon // Small bias to ensure separation
                    };
                }

                // Insert the new vertex between the closest edge's endpoints
                polytope.Insert(closestEdge.IndexB, newSupportVertex);
            }

            // Fallback if max iterations reached
            PolytopeEdge fallbackEdge = FindClosestEdge(polytope);
            return new CollisionResult
            {
                IsColliding = true,
                PenetrationNormal = fallbackEdge.Normal,
                PenetrationDepth = fallbackEdge.Distance
            };
        }

        ///// <summary>
        ///// Public method to test collision between two convex polygons.
        ///// Returns detailed collision information (normal and depth) if colliding.
        ///// </summary>
        ///// <param name="verticesA">Vertices of first convex polygon (counterclockwise)</param>
        ///// <param name="verticesB">Vertices of second convex polygon (counterclockwise)</param>
        ///// <returns>CollisionResult structure</returns>
        //public static CollisionResult DetectCollision(
        //    IReadOnlyList<Vector2> verticesA,
        //    IReadOnlyList<Vector2> verticesB)
        //{
        //    if (GilbertJohnsonKeerthiAlgorithm(verticesA, verticesB, out List<MinkowskiVertex> simplex))
        //    {
        //        if (simplex == null || simplex.Count < 3)
        //        {
        //            // Deep penetration or degenerate case – treat as colliding with zero vector
        //            return new CollisionResult { IsColliding = true, PenetrationNormal = Vector2.Zero, PenetrationDepth = 0f };
        //        }

        //        return ExpandingPolytopeAlgorithm(verticesA, verticesB, simplex);
        //    }

        //    return new CollisionResult { IsColliding = false };
        //}


        /// <summary>
        /// Detects collision between two single convex polygons.
        /// </summary>
        public static CollisionResult DetectCollision(
            IReadOnlyList<Vector2D> convexVerticesA,
            IReadOnlyList<Vector2D> convexVerticesB)
        {
            if (GilbertJohnsonKeerthiAlgorithm(convexVerticesA, convexVerticesB, out List<MinkowskiVertex> simplex))
            {
                if (simplex == null || simplex.Count < 3)
                {
                    // Deep penetration or degenerate case
                    return new CollisionResult
                    {
                        IsColliding = true,
                        PenetrationNormal = Vector2D.Zero,
                        PenetrationDepth = 0
                    };
                }

                return ExpandingPolytopeAlgorithm(convexVerticesA, convexVerticesB, simplex);
            }

            return new CollisionResult { IsColliding = false };
        }

        /// <summary>
        /// Detects collision between two compound (possibly non-convex) shapes.
        /// Each shape is represented as a collection of convex polygons (its decomposition).
        /// Returns the deepest penetration among all colliding pairs.
        /// If no collision, returns IsColliding = false.
        /// </summary>
        /// <param name="compoundShapeA">Collection of convex polygons making up shape A</param>
        /// <param name="compoundShapeB">Collection of convex polygons making up shape B</param>
        /// <returns>CollisionResult with the deepest penetration (or no collision)</returns>
        public static CollisionResult DetectCollision(
            IReadOnlyCollection<IReadOnlyList<Vector2D>> compoundShapeA,
            IReadOnlyCollection<IReadOnlyList<Vector2D>> compoundShapeB)
        {
            if (compoundShapeA == null || compoundShapeB == null)
                throw new ArgumentNullException();
            if (compoundShapeA.Count == 0 || compoundShapeB.Count == 0)
                return new CollisionResult { IsColliding = false };

            CollisionResult bestResult = default;
            bool foundCollision = false;

            foreach (var convexA in compoundShapeA)
            {
                foreach (var convexB in compoundShapeB)
                {
                    CollisionResult result = DetectCollision(convexA, convexB);
                    if (result.IsColliding)
                    {
                        if (!foundCollision || result.PenetrationDepth > bestResult.PenetrationDepth + Epsilon)
                        {
                            bestResult = result;
                        }
                        foundCollision = true;
                    }
                }
            }

            return foundCollision ? bestResult : new CollisionResult { IsColliding = false };
        }

        public static CollisionResult DetectCollision(Polygon2D convexA, Polygon2D convexB) =>
            DetectCollision(
                convexA.Vertices,
                convexB.Vertices
            );

        public static CollisionResult DetectCollision(Polygon2D[] compoundShapeA, Polygon2D[] compoundShapeB) =>
            DetectCollision(
                [.. compoundShapeA.Select(p => p.Vertices)],
                [.. compoundShapeB.Select(p => p.Vertices)]
            );
    }

}
