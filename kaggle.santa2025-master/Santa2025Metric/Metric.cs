using kaggle.santa2025.ChristmasTrees;
using NetTopologySuite.Geometries;
using NetTopologySuite.Index.Strtree;
using NetTopologySuite.Operation.Union;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace kaggle.santa2025.Santa2025Metric
{
    public static class Metric
    {
        public static double GetScore(IList<Geometry> polygons)
        {
            var strTree = new STRtree<Geometry>();
            for (int i = 0; i < polygons.Count; i++)
            {
                strTree.Insert(polygons[i].EnvelopeInternal, polygons[i]);
            }
            strTree.Build();

            for (int i = 0; i < polygons.Count; i++)
            {
                Geometry poly = polygons[i];
                IList<Geometry> candidates = strTree.Query(poly.EnvelopeInternal);

                foreach (var candidate in candidates)
                {
                    if (ReferenceEquals(candidate, poly)) continue;
                    if (poly.Intersects(candidate) && !poly.Touches(candidate)) return -1;
                }
            }

            Geometry union = UnaryUnionOp.Union(polygons);
            Envelope env = union.EnvelopeInternal;

            double sideLength = Math.Max(env.Width, env.Height);
            return (sideLength * sideLength) / polygons.Count;
        }

        public static double GetScore(IEnumerable<ChristmasTree> christmasTrees)
        {
            var polygons = new List<Geometry>();
            foreach (ChristmasTree christmasTree in christmasTrees)
            {
                polygons.Add(new MetricTree(christmasTree.Translation.X, christmasTree.Translation.Y, christmasTree.TemplateIndex).Polygon);
            }
            return GetScore(polygons);
        }

        public static double GetScore(string submission)
        {
            if (string.IsNullOrWhiteSpace(submission))
            {
                Console.WriteLine("Error: Submission cannot be empty.");
                return -1;
            }

            var lines = submission.Split([ '\r', '\n' ], StringSplitOptions.RemoveEmptyEntries);
            var rows = new List<(string id, double x, double y, double deg)>();

            const double limit = 100.0;

            foreach (var line in lines)
            {
                if (line == "id,x,y,deg") continue;

                var parts = line.Split(',');
                if (parts.Length != 4)
                {
                    Console.WriteLine($"Error: Invalid line (expected 4 columns): {line}");
                    return -1;
                }

                string id = parts[0].Trim();
                string xStr = parts[1].Trim();
                string yStr = parts[2].Trim();
                string degStr = parts[3].Trim();

                if (!xStr.StartsWith("s") || !yStr.StartsWith("s") || !degStr.StartsWith("s"))
                {
                    Console.WriteLine($"Error: Invalid line (expected 's' prefix): {line}");
                    return -1;
                }

                double x = double.Parse(xStr.AsSpan(1), CultureInfo.InvariantCulture);
                double y = double.Parse(yStr.AsSpan(1), CultureInfo.InvariantCulture);
                double deg = double.Parse(degStr.AsSpan(1), CultureInfo.InvariantCulture);

                if (Math.Abs(x) > limit || Math.Abs(y) > limit)
                {
                    Console.WriteLine($"Error: x or y value out of bounds (-100 to 100): {line}");
                    return -1;
                }

                rows.Add((id, x, y, deg));
            }

            var groups = rows.GroupBy(r => r.id.Split('_')[0]);

            decimal totalScore = 0m;

            foreach (var group in groups)
            {
                var groupRows = group.ToList();
                int numTrees = groupRows.Count;

                var polygons = new List<Geometry>();

                foreach (var row in groupRows)
                {
                    var tree = new MetricTree(row.x, row.y, row.deg);
                    polygons.Add(tree.Polygon);
                }

                double groupScore = GetScore(polygons);
                if (groupScore < 0)
                {
                    Console.WriteLine($"Error: Overlapping trees detected in group {group.Key}");
                    return -1;
                }

                totalScore += (decimal)groupScore;
            }

            return (double)totalScore;
        }
    }
}
