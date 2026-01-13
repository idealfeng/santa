using kaggle.santa2025.ChristmasTrees;
using kaggle.santa2025.Geometry2D;
using System;
using System.Linq;
using System.Text;

namespace kaggle.santa2025.Packing
{
    public class Placement
    {
        public ChristmasTree[] Trees = null;
        public double SideLength => ComputeSideLength();

        private double ComputeSideLength()
        {
            if (Trees.Length == 0) return 0;

            double minX = double.MaxValue, minY = double.MaxValue;
            double maxX = double.MinValue, maxY = double.MinValue;

            foreach (var tree in Trees)
            {
                var aabb = ChristmasTreeFactory.GetAABB(tree);
                minX = Math.Min(minX, aabb.Min.X);
                minY = Math.Min(minY, aabb.Min.Y);
                maxX = Math.Max(maxX, aabb.Max.X);
                maxY = Math.Max(maxY, aabb.Max.Y);
            }

            double side = Math.Max(maxX - minX, maxY - minY);
            return side;
        }

        public Placement Clone()
        {
            return new Placement
            {
                Trees = [.. Trees.Select(t => ChristmasTreeFactory.Create(t.Translation, t.TemplateIndex))]
            };
        }

        public Placement Clone(Vector2D delta)
        {
            return new Placement
            {
                Trees = [.. Trees.Select(t => ChristmasTreeFactory.Create(t.Translation + delta, t.TemplateIndex))]
            };
        }

        public Placement CloneAdjusted(out double side)
        {
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            foreach (var tree in Trees)
            {
                var aabb = ChristmasTreeFactory.GetAABB(tree);
                minX = Math.Min(minX, aabb.Min.X);
                minY = Math.Min(minY, aabb.Min.Y);
                maxX = Math.Max(maxX, aabb.Max.X);
                maxY = Math.Max(maxY, aabb.Max.Y);
            }

            side = Math.Max(maxX - minX, maxY - minY);
            return Clone(new Vector2D(-minX, -minY));
        }

        public double GetScore()
        {
            return Santa2025Metric.Metric.GetScore(Trees);
        }

        public string ExportSvg()
        {
            Placement layout = this.CloneAdjusted(out double side);

            StringBuilder sb = new();
            sb.AppendLine($"<svg viewBox=\"0 0 {side + 2} {side + 2}\" xmlns=\"http://www.w3.org/2000/svg\">");
            sb.AppendLine($"  <rect x=\"0\" y=\"0\" width=\"{side}\" height=\"{side}\" fill=\"none\" stroke=\"#333\" stroke-width=\"0.05\"/>");

            foreach (ChristmasTree tree in layout.Trees)
            {
                Polygon2D polygon = ChristmasTreeFactory.TEMPLATE_POLYGONS[tree.TemplateIndex].Translated(tree.Translation);
                string points = string.Join(" ", polygon.Vertices.Select(v => $"{v.X:F4},{v.Y:F4}"));
                sb.AppendLine($"  <polygon points=\"{points}\" fill=\"#4488ff\" opacity=\"0.7\" stroke=\"black\" stroke-width=\"0.02\"/>");
            }
            sb.AppendLine("</svg>");
            return sb.ToString();
        }

        public StringBuilder ExportPlacementSolution(StringBuilder sb)
        {
            int n = Trees.Length;
            for (int i = 0; i < n; i++)
            {
                (Vector2D root, _) = ChristmasTreeFactory.GetRootAndAngle(Trees[i]);
                sb.AppendLine($"{n:D3}_{i},s{root.X:0.0#},s{root.Y:0.0#},s{Trees[i].TemplateIndex:0.0#}");
            }
            return sb;
        }
    }
}
