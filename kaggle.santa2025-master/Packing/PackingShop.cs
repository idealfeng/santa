using kaggle.santa2025.ChristmasTrees;
using kaggle.santa2025.CollisionDetection2D;
using kaggle.santa2025.Geometry2D;
using System;

namespace kaggle.santa2025.Packing
{
    public static class PackingShop
    {
        const int NUM_OF_ATTEMPTS = 50;
        static readonly TimeSpan TIME_LIMIT = TimeSpan.FromSeconds(10);

        static readonly Random random = new(2025);

        public static Placement Generate(int numTrees, double spread)
        {
            spread = spread * Math.Sqrt(numTrees);

            for (int attempt = 0; attempt < NUM_OF_ATTEMPTS; attempt++)
            {
                DateTime startTime = DateTime.Now;
                bool timeout = false;

                ChristmasTree[] trees = new ChristmasTree[numTrees];
                for (int i = 0; i < numTrees; i++)
                {
                    int T = 0;
                    while (true)
                    {
                        ChristmasTree newTree = ChristmasTreeFactory.Create(new Vector2D(random.NextDouble() * spread, random.NextDouble() * spread), random.Next(0, 360));
                        T++;
                        bool hasCollision = false;
                        for (int j = 0; j < i; j++)
                        {
                            CollisionResult collision = ChristmasTreeFactory.DetectCollision(newTree, trees[j]);
                            if (collision.IsColliding)
                            {
                                hasCollision = true;
                                break;
                            }
                        }
                        if (!hasCollision)
                        {
                            trees[i] = newTree;
                            startTime = DateTime.Now;
                            break;
                        }
                        if (DateTime.Now - startTime > TIME_LIMIT)
                        {
                            timeout = true;
                            Console.WriteLine($"Attempt {attempt + 1}: Timeout {T} reached after placing {i} trees.");
                            break;
                        }
                    }
                    if (timeout) break;
                }
                if (timeout) continue;

                Console.WriteLine("Initial placement generated.");

                return new Placement
                {
                    Trees = trees
                };
            }

            return null;
        }
    }
}
