using System;
using System.Linq;

namespace kaggle.santa2025.Geometry2D
{
    public readonly struct Polygon2D(Vector2D[] vertices)
    {
        public Vector2D[] Vertices { get; init; } = vertices?.ToArray() ?? throw new ArgumentNullException(nameof(vertices));

        public readonly int VertexCount => Vertices.Length;

        public readonly double Area()
        {
            double area = 0.0;
            int n = Vertices.Length;
            for (int i = 0; i < n; i++)
            {
                Vector2D v1 = Vertices[i];
                Vector2D v2 = Vertices[(i + 1) % n];
                area += (v1.X * v2.Y) - (v2.X * v1.Y);
            }
            return Math.Abs(area) * 0.5;
        }

        public readonly Vector2D Centroid()
        {
            double cx = 0, cy = 0;
            foreach (var v in Vertices)
            {
                cx += v.X;
                cy += v.Y;
            }
            return new Vector2D(cx / Vertices.Length, cy / Vertices.Length);
        }

        public Polygon2D Translated(Vector2D delta)
        {
            Vector2D[] newVertices = new Vector2D[Vertices.Length];
            for (int i = 0; i < Vertices.Length; i++)
            {
                newVertices[i] = Vertices[i] + delta;
            }
            return new Polygon2D(newVertices);
        }

        public Polygon2D Translated(double dx, double dy) => Translated(new Vector2D(dx, dy));

        public Polygon2D Rotated(double angleRadians)
        {
            Vector2D[] newVertices = new Vector2D[Vertices.Length];
            for (int i = 0; i < Vertices.Length; i++)
            {
                newVertices[i] = Vertices[i].Rotated(angleRadians);
            }
            return new Polygon2D(newVertices);
        }

        public Polygon2D Scaled(double scaleX, double scaleY)
        {
            Vector2D[] newVertices = new Vector2D[Vertices.Length];
            for (int i = 0; i < Vertices.Length; i++)
            {
                newVertices[i] = Vertices[i].Scaled(scaleX, scaleY);
            }
            return new Polygon2D(newVertices);
        }

        public Polygon2D Scaled(double uniformScale) => Scaled(uniformScale, uniformScale);

        public void WithTranslated(Vector2D delta, Span<Vector2D> destination)
        {
            for (int i = 0; i < Vertices.Length; i++)
            {
                destination[i] = Vertices[i] + delta;
            }
        }

        public readonly Polygon2D Inflated(double distance)
        {
            if (Vertices.Length < 3) return this; // degenerate
            if (distance < 1e-9) return this; // outward only

            Vector2D[] offsetVertices = new Vector2D[Vertices.Length];
            int n = Vertices.Length;

            for (int i = 0; i < n; i++)
            {
                Vector2D prev = Vertices[(i - 1 + n) % n];
                Vector2D curr = Vertices[i];
                Vector2D next = Vertices[(i + 1) % n];

                Vector2D dirPrev = (curr - prev).Normalized();
                Vector2D dirNext = (next - curr).Normalized();

                // Outward normal (rotate 90° CCW)
                Vector2D normalPrev = new(-dirPrev.Y, dirPrev.X);
                Vector2D normalNext = new(-dirNext.Y, dirNext.X);

                // For outward offset, average the normals at vertex for better miter
                Vector2D bisector = (normalPrev + normalNext).Normalized() * distance;

                offsetVertices[i] = curr + bisector;
            }

            return new Polygon2D(offsetVertices);
        }

        public override string ToString() => $"Polygon2D({VertexCount} vertices)";
    }
}
