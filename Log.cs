namespace PerceptronTR;

internal class Log
{
    record Tag(string Text, ConsoleColor Color);

    static readonly Tag MatchTag = new($"{nameof(Match)}      ", ConsoleColor.Green);
    static readonly Tag MismatchTag = new($"{nameof(Mismatch)}   ", ConsoleColor.Red);
    static readonly Tag BackwardingTag = new(nameof(Backwarding), ConsoleColor.Blue);

    static void LogWithTag(Tag tag, string message)
    {
        var originalColor = Console.ForegroundColor;
        {
            Console.ForegroundColor = tag.Color;
            Console.Write($"{tag.Text} ");
        }
        Console.ForegroundColor = originalColor;
        Console.WriteLine(message);
    }

    public static void Match(string message) => LogWithTag(MatchTag, message);
    public static void Mismatch(string message) => LogWithTag(MismatchTag, message);
    public static void Backwarding(string message) => LogWithTag(BackwardingTag, message);
}
