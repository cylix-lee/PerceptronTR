namespace PerceptronTR;

interface IPerceptronDataset
{
    int GetItemCount();
    (IEnumerable<int>, int) GetItem(int index);
}

static class PerceptronDatasetExtension
{
    public static IEnumerable<(IEnumerable<int>, int)> GetItems(this IPerceptronDataset dataset)
    {
        var items = new List<(IEnumerable<int>, int)>();
        for (var i = 0; i < dataset.GetItemCount(); i++)
        {
            items.Add(dataset.GetItem(i));
        }

        return items;
    }
}