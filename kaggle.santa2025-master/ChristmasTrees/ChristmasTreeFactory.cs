using kaggle.santa2025.CollisionDetection2D;
using kaggle.santa2025.Geometry2D;
using System;

namespace kaggle.santa2025.ChristmasTrees
{
    public static class ChristmasTreeFactory
    {
        public const int NUM_OF_TEMPLATES = 360;
        public const int NUM_OF_TREE_PARTS = 4;

        // counter-clockwise order
        static readonly Polygon2D BASE_POLYGON = new ([
            new(0.0, 0.5),
            new(-0.125, 0.2), new(-0.0625, 0.2),
            new(-0.2, -0.05), new(-0.1, -0.05),
            new(-0.35, -0.3), new(-0.075, -0.3),
            new(-0.075, -0.5),
            new(0.075, -0.5),
            new(0.075, -0.3),new(0.35, -0.3),
            new(0.1, -0.05), new(0.2, -0.05),
            new(0.0625, 0.2), new(0.125, 0.2),
        ]);

        // counter-clockwise order
        static readonly Polygon2D[] BASE_TREE = [
            new Polygon2D([
                new(0.0, 0.5),
                new(-0.125, 0.2), new(0.125, 0.2),
            ]),
            new Polygon2D([
                new(-0.0625, 0.2), new(-0.2, -0.05),
                new(0.2, -0.05), new(0.0625, 0.2),
            ]),
            new Polygon2D([
                new(-0.1, -0.05), new(-0.35, -0.3),
                new(0.35, -0.3), new(0.1, -0.05),
            ]),
            new Polygon2D([
                new(-0.075, -0.3),
                new(-0.075, -0.5),
                new(0.075, -0.5),
                new(0.075, -0.3),
            ]),
        ];

        public static readonly double BASE_POLYGON_AREA;

        static readonly Vector2D BASE_ROOT = new(0.0, -0.3);

        public static readonly Polygon2D[] TEMPLATE_POLYGONS;
        public static readonly Polygon2D[][] TEMPLATE_TREES;
        static readonly Vector2D[] TEMPLATE_ROOTS;
        static readonly AABB[] TEMPLATE_AABBS;

        static ChristmasTreeFactory()
        {
            BASE_POLYGON_AREA = BASE_POLYGON.Area();

            TEMPLATE_POLYGONS = new Polygon2D[NUM_OF_TEMPLATES];
            TEMPLATE_TREES = new Polygon2D[NUM_OF_TEMPLATES][];
            TEMPLATE_ROOTS = new Vector2D[NUM_OF_TEMPLATES];
            TEMPLATE_AABBS = new AABB[NUM_OF_TEMPLATES];

            for (int i = 0; i < NUM_OF_TEMPLATES; i++)
            {
                double angle = i * Math.PI / 180.0;
                TEMPLATE_POLYGONS[i] = BASE_POLYGON.Rotated(angle);
                TEMPLATE_TREES[i] = new Polygon2D[NUM_OF_TREE_PARTS];
                for (int j = 0; j < NUM_OF_TREE_PARTS; j++)
                {
                    TEMPLATE_TREES[i][j] = BASE_TREE[j].Rotated(angle);
                }
                TEMPLATE_ROOTS[i] = BASE_ROOT.Rotated(angle);
                TEMPLATE_AABBS[i] = AABB.FromPoints(TEMPLATE_POLYGONS[i].Vertices);
            }
        }

        public static AABB GetAABB(ChristmasTree tree)
        {
            return TEMPLATE_AABBS[tree.TemplateIndex].Translated(tree.Translation);
        }

        public static ChristmasTree Create(Vector2D translation, int templateIndex)
        {
            Polygon2D[] templateTree = TEMPLATE_TREES[templateIndex];
            Polygon2D[] tree = new Polygon2D[NUM_OF_TREE_PARTS];
            for (int i = 0; i < NUM_OF_TREE_PARTS; i++)
            {
                tree[i] = templateTree[i].Translated(translation);
            }
            return new ChristmasTree(translation, templateIndex, tree);
        }

        public static (Vector2D root, double angle) GetRootAndAngle(Vector2D translation, int templateIndex)
        {
            return (TEMPLATE_ROOTS[templateIndex].Translated(translation), templateIndex * Math.PI / 180.0);
        }

        public static (Vector2D root, double angle) GetRootAndAngle(ChristmasTree tree)
        {
            return GetRootAndAngle(tree.Translation, tree.TemplateIndex);
        }

        public static bool CheckAABBOverlap(ChristmasTree a, ChristmasTree b)
        {
            AABB aabbA = TEMPLATE_AABBS[a.TemplateIndex].Translated(a.Translation);
            AABB aabbB = TEMPLATE_AABBS[b.TemplateIndex].Translated(b.Translation);
            return aabbA.Overlaps(aabbB);
        }

        public static CollisionResult DetectCollision(ChristmasTree a, ChristmasTree b)
        {
            if (!CheckAABBOverlap(a, b))
            {
                return new CollisionResult { IsColliding = false };
            }
            return GjkEpaCollisionDetector.DetectCollision(a.Tree, b.Tree);
        }
    }
}
