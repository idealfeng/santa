using kaggle.santa2025.ChristmasTrees;
using kaggle.santa2025.Geometry2D;
using System;
using System.Collections.Generic;

namespace kaggle.santa2025.Packing
{
    public static class GravityPacker
    {
        private const double GravityStep = 0.05;        // How far to try moving down each iteration
        private const double ResolutionPushFactor = 1.1; // Slightly over-correct penetrations
        private const double JitterStrength = 0.3;
        private const int MaxIterations = 20000;
        private const int JitterEvery = 300;            // Add random shake periodically

        static readonly Random random = new(2025);

        public static Placement Pack(Placement initial)
        {
            var placement = initial.Clone();

            for (int iter = 0; iter < MaxIterations; iter++)
            {
                bool moved = false;
                int n = placement.Trees.Length;

                // Phase 1: Apply gravity and resolve collisions
                for (int i = 0; i < n; i++)
                {
                    ChristmasTree currentTree = placement.Trees[i];
                    Vector2D originalPos = currentTree.Translation;

                    // Try moving down
                    Vector2D testPos = originalPos + new Vector2D(0, -GravityStep);

                    // Temporarily update position (we'll rebuild the tree)
                    ChristmasTree movedTree = ChristmasTreeFactory.Create(testPos, currentTree.TemplateIndex);

                    // Collect total push from all collisions
                    Vector2D totalPush = Vector2D.Zero;
                    bool hasCollision = false;

                    for (int j = 0; j < n; j++)
                    {
                        if (i == j) continue; // Skip self

                        var result = ChristmasTreeFactory.DetectCollision(movedTree, placement.Trees[j]);
                        if (result.IsColliding && result.PenetrationDepth > 1e-6)
                        {
                            hasCollision = true;
                            // Normal points from trees[j] to movedTree → push movedTree away
                            totalPush += result.PenetrationNormal * (result.PenetrationDepth * ResolutionPushFactor);
                        }
                    }

                    Vector2D finalPos;
                    if (hasCollision)
                    {
                        finalPos = originalPos + totalPush;
                    }
                    else
                    {
                        finalPos = testPos; // Successfully moved down
                        moved = true;
                    }

                    // Only rebuild and replace if position actually changed
                    if ((finalPos - placement.Trees[i].Translation).LengthSquared() > 1e-8)
                    {
                        placement.Trees[i] = ChristmasTreeFactory.Create(finalPos, currentTree.TemplateIndex);
                        moved = true;
                    }
                }

                // Phase 2: Occasional jitter
                if (iter % JitterEvery == 0 && iter > 0)
                {
                    //ApplyJitter(placement.Trees, JitterStrength);
                }

                // Optional: stronger jitter if completely stuck
                if (iter > 5000 && !moved && iter % 500 == 0)
                {
                    //ApplyJitter(placement.Trees, JitterStrength * 4);
                }
            }

            return placement;
        }

        private static void ApplyJitter(List<ChristmasTree> trees, double strength)
        {
            int n = trees.Count;
            for (int i = 0; i < n; i++)
            {
                ChristmasTree tree = trees[i];

                double dx = (random.NextDouble() - 0.5) * strength * 2;
                double dy = (random.NextDouble() - 0.5) * strength * 2;
                Vector2D newPos = tree.Translation + new Vector2D(dx, dy);

                int newRotation = tree.TemplateIndex;

                // Occasionally change rotation
                if (random.NextDouble() < 0.2) // 20% chance per jitter
                {
                    int delta = random.Next(-15, 16); // ±15 degrees
                    newRotation = (tree.TemplateIndex + delta + 360) % 360;
                }

                // Rebuild tree with new pos/rotation
                trees[i] = ChristmasTreeFactory.Create(newPos, newRotation);
            }
        }
    }
}
