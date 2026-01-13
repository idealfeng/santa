using System;

namespace kaggle.santa2025.Geometry2D
{
    public readonly struct AABB(Vector2D min, Vector2D max)
    {
        public Vector2D Min { get; init; } = min;
        public Vector2D Max { get; init; } = max;

        public readonly Vector2D Size => Max - Min;

        public readonly Vector2D Center => (Min + Max) * 0.5;

        public static AABB FromCenter(Vector2D center, Vector2D halfSize) => new(center - halfSize, center + halfSize);

        public AABB Translated(Vector2D delta) => new(Min + delta, Max + delta);

        public readonly bool Overlaps(AABB other) =>
            Min.X <= other.Max.X && Max.X >= other.Min.X &&
            Min.Y <= other.Max.Y && Max.Y >= other.Min.Y;

        public readonly bool Contains(Vector2D point) =>
                point.X >= Min.X && point.X <= Max.X &&
                point.Y >= Min.Y && point.Y <= Max.Y;

        public readonly AABB Union(AABB other) => new(
                Vector2D.Min(Min, other.Min),
                Vector2D.Max(Max, other.Max));

        public static AABB FromPoints(Vector2D[] vertices)
        {
            if (vertices.Length == 0)
                return new AABB(Vector2D.Zero, Vector2D.Zero); // or throw?

            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            foreach (var v in vertices)
            {
                minX = Math.Min(minX, v.X);
                maxX = Math.Max(maxX, v.X);
                minY = Math.Min(minY, v.Y);
                maxY = Math.Max(maxY, v.Y);
            }

            return new AABB(new(minX, minY), new(maxX, maxY));
        }

        public override string ToString() => $"AABB[{Min} - {Max}]";
    }
}
