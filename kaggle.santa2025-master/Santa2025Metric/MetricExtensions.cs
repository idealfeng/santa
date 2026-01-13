using NetTopologySuite.Geometries;
using System;

namespace kaggle.santa2025.Santa2025Metric
{
    public static class MetricExtensions
    {
        public static Geometry Rotate(this Geometry geom, double originX, double originY, double degrees)
        {
            var radians = degrees * Math.PI / 180.0;
            var cos = Math.Cos(radians);
            var sin = Math.Sin(radians);

            if (geom is Polygon polygon)
            {
                var newExterior = RotateRing(polygon.ExteriorRing.Coordinates, originX, originY, cos, sin);
                var newHoles = new LinearRing[polygon.NumInteriorRings];

                for (int i = 0; i < polygon.NumInteriorRings; i++)
                {
                    var holeCoords = polygon.GetInteriorRingN(i).Coordinates;
                    newHoles[i] = geom.Factory.CreateLinearRing(RotateRing(holeCoords, originX, originY, cos, sin));
                }

                return geom.Factory.CreatePolygon(geom.Factory.CreateLinearRing(newExterior), newHoles);
            }

            throw new NotSupportedException("Rotation only implemented for Polygon.");
        }

        private static Coordinate[] RotateRing(Coordinate[] coords, double ox, double oy, double cos, double sin)
        {
            var result = new Coordinate[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                double dx = coords[i].X - ox;
                double dy = coords[i].Y - oy;
                result[i] = new Coordinate(ox + dx * cos - dy * sin, oy + dy * cos + dx * sin);
            }
            return result;
        }

        public static Geometry Translate(this Geometry geom, double dx, double dy)
        {
            if (geom is Polygon polygon)
            {
                var newExterior = TranslateRing(polygon.ExteriorRing.Coordinates, dx, dy);
                var newHoles = new LinearRing[polygon.NumInteriorRings];

                for (int i = 0; i < polygon.NumInteriorRings; i++)
                {
                    var hole = polygon.GetInteriorRingN(i);
                    newHoles[i] = geom.Factory.CreateLinearRing(TranslateRing(hole.Coordinates, dx, dy));
                }

                return geom.Factory.CreatePolygon(geom.Factory.CreateLinearRing(newExterior), newHoles);
            }

            throw new NotSupportedException("Translation only implemented for Polygon.");
        }

        private static Coordinate[] TranslateRing(Coordinate[] coords, double dx, double dy)
        {
            var result = new Coordinate[coords.Length];
            for (int i = 0; i < coords.Length; i++)
                result[i] = new Coordinate(coords[i].X + dx, coords[i].Y + dy);
            return result;
        }
    }
}
