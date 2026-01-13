using kaggle.santa2025.Geometry2D;
using System;
using System.Linq;

namespace kaggle.santa2025.ChristmasTrees
{
    public readonly struct ChristmasTree(Vector2D translation, int templateIndex, Polygon2D[] treeParts)
    {
        public Vector2D Translation { get; init; } = translation;
        public int TemplateIndex { get; init; } = templateIndex;
        public Polygon2D[] Tree { get; init; } = treeParts?.ToArray() ?? throw new ArgumentNullException(nameof(treeParts));
    }
}
