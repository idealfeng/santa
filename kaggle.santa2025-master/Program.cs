using kaggle.santa2025.Packing;
using System;
using System.Text;

namespace kaggle.santa2025
{
    public class Program
    {
        static string LOG_FILE = $@"c:\temp\santa\output\log_{DateTime.Now:yyyyMMdd}_{DateTime.Now:HHmmss}.txt";

        static void Log(string message)
        {
            Console.WriteLine(message);
            System.IO.File.AppendAllText(LOG_FILE, $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}   {message}{Environment.NewLine}");
        }

        static void Main(string[] args)
        {
            Log("Started...");

            StringBuilder sb = new ();
            sb.AppendLine("id,x,y,deg");

            double spread = 0.41;
            double totalScore = 0;

            for (int k = 1; k <= 200; k++)
            {
                Log($"Placing {k} trees with initial spread {spread}...");
                while (true)
                {
                    Placement layout = PackingShop.Generate(k, spread);
                    if (layout == null)
                    {
                        spread *= 1.005;
                        Log($"Increasing spread to {spread}...");
                        continue;
                    }

                    double score = layout.GetScore();
                    if (score > 0)
                    {
                        Console.WriteLine($"{k}/200 Score: {score}");
                        layout.ExportPlacementSolution(sb);
                        totalScore += score;

                        System.IO.File.WriteAllText($@"c:\temp\santa\output\santa2025_{DateTime.Now:yyyyMMdd_HHmmss}_{k}.svg", layout.ExportSvg());
                        break;
                    }
                }
            }

            System.IO.File.WriteAllText($@"c:\temp\santa\output\santa2025_{DateTime.Now:yyyyMMdd_HHmmss}.csv", sb.ToString());
            Log($"Total score: {totalScore})");
            Log("Done!");
        }
    }
}
