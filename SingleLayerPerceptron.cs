namespace PerceptronTR;

class SingleLayerPerceptron
{
    readonly double[] weights;
    double threshold;

    public SingleLayerPerceptron(int inputDimension)
    {
        weights = new double[inputDimension];
        for (var i = 0; i < inputDimension; i++)
        {
            // Random initializing weights, ranging from 0.0 to 1.0.
            weights[i] = Random.Shared.NextDouble();
        }
        threshold = Random.Shared.NextDouble();
        Console.WriteLine($"Initialized perceptron with weights {weights.CollectAsString()}, threshold {threshold}");
    }

    public int Forward(IEnumerable<int> input)
    {
        // Check input validity
        if (input.Count() != weights.Length)
        {
            throw new ApplicationException($"Unacceptable input dimension: " +
                $"expected {weights.Length}, got {input.Count()}");
        }

        // Calculation: multiply input with corresponding weights, and minus threshold at last.
        double calculation = 0;
        for (var j = 0; j < weights.Length; j++)
        {
            calculation += input.ElementAt(j) * weights[j];
        }
        calculation -= threshold;

        // Pass the calculation result to activation function, and return output.
        return calculation > 0 ? 1 : 0;
    }

    void Backward(IEnumerable<int> input, int output, int groundTruth, double learningRate)
    {
        // Calculate difference and update weights and threshold.
        var difference = groundTruth - output;
        for (var i = 0; i < weights.Length; i++)
        {
            weights[i] += learningRate * difference * input.ElementAt(i);
        }
        threshold -= learningRate * difference;
    }

    public void Fit(IPerceptronDataset dataset, int? epoch = null, double learningRate = 0.1)
    {
        if (epoch is null) // Representing fitting stops only when the model is totally fitted.
        {
            bool isFitted = false;
            while (!isFitted)
            {
                isFitted = true;
                foreach (var (input, groundTruth) in dataset.GetItems())
                {
                    var output = Forward(input);
                    if (output == groundTruth)
                        Log.Match($"{input.CollectAsString()} \t expected {groundTruth}, got {output}");
                    else
                    {
                        isFitted = false;
                        Log.Mismatch($"{input.CollectAsString()} \t expected {groundTruth}, got {output}");
                        Backward(input, output, groundTruth, learningRate);
                        Log.Backwarding($"new weights {weights.CollectAsString()}, new threshold {threshold}");
                    }
                }
            }
        }
        else // Fitting will stop according to specified epoch.
        {
            for (var i = 0; i < epoch; i++)
            {
                foreach (var (input, groundTruth) in dataset.GetItems())
                {
                    var output = Forward(input);
                    if (output == groundTruth)
                        Log.Match($"{input.CollectAsString()} \t expected {groundTruth}, got {output}");
                    else
                    {
                        Log.Mismatch($"{input.CollectAsString()} \t expected {groundTruth}, got {output}");
                        Backward(input, output, groundTruth, learningRate);
                        Log.Backwarding($"new weights {weights.CollectAsString()}, new threshold {threshold}");
                    }
                }
            }
        }
        Console.WriteLine($"Fitting stopped with weights {weights.CollectAsString()}, threshold {threshold}");
        Console.WriteLine();
    }

    public void TestAccuracy(IPerceptronDataset dataset)
    {
        Console.WriteLine("Testing model accuracy...");

        var totalCases = dataset.GetItemCount();
        var correctCases = 0;
        foreach (var (input, groundTruth) in dataset.GetItems())
        {
            var output = Forward(input);
            if (output == groundTruth)
            {
                Log.Match($"{input.CollectAsString()} \t expected {groundTruth}, got {output}");
                correctCases++;
            }
            else Log.Mismatch($"{input.CollectAsString()} \t expected {groundTruth}, got {output}");
        }
        Console.WriteLine($"Testing stopped with model accuracy {((double)correctCases / totalCases) * 100}%");
    }
}
