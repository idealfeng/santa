using NetTopologySuite.Geometries;

namespace kaggle.santa2025.Santa2025Metric
{
    public class MetricTree
    {
        public Geometry Polygon { get; }
        private static readonly GeometryFactory Factory = new();

        public MetricTree(double centerX, double centerY, double angleDegrees)
        {
            var points = new Coordinate[]
            {
                new(0.0, 0.8),   // tip

                new(0.25 / 2, 0.5),
                new(0.25 / 4, 0.5),

                new(0.4 / 2,  0.25),
                new(0.4 / 4,  0.25),

                new(0.7 / 2,  0.0),

                new(0.15 / 2, 0.0),
                new(0.15 / 2, -0.2),

                new(-0.15 / 2, -0.2),
                new(-0.15 / 2, 0.0),

                new(-0.7 / 2,  0.0),

                new(-0.4 / 4,  0.25),
                new(-0.4 / 2,  0.25),

                new(-0.25 / 4, 0.5),
                new(-0.25 / 2, 0.5),

                new(0.0, 0.8)    // close ring
            };

            var ring = Factory.CreateLinearRing(points);
            var initialPolygon = Factory.CreatePolygon(ring);

            var rotated = initialPolygon.Rotate(0, 0, angleDegrees);
            var translated = rotated.Translate(centerX, centerY);

            Polygon = translated;
        }
    }
}
