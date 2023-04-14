using System.Text;
using PerceptronTR.Datasets;

namespace PerceptronTR;

class Program
{
    static void Main(string[] args)
    {
        var dataset = new AndDataset();
        var perceptron = new SingleLayerPerceptron(2);

        perceptron.Fit(dataset);
        Console.WriteLine();
        perceptron.TestAccuracy(dataset);
    }
}

static class ToStringExtension
{
    public static string CollectAsString<T>(this IEnumerable<T> items) where T : notnull
    {
        var stringBuilder = new StringBuilder();
        foreach (var item in items)
        {
            stringBuilder.Append(item.ToString());
            stringBuilder.Append(' ');
        }

        return stringBuilder.ToString();
    }
}