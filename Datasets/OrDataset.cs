namespace PerceptronTR.Datasets;

class OrDataset : IPerceptronDataset
{
    readonly IEnumerable<int>[] feature =
    {
        new int[] {0, 0},
        new int[] {0, 1},
        new int[] {1, 0},
        new int[] {1, 1},
    };
    readonly int[] groundTruth = { 0, 1, 1, 1 };

    public int GetItemCount() => feature.Length;
    public (IEnumerable<int>, int) GetItem(int index) => (feature[index], groundTruth[index]);
}
