import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.HashSet;

public class UniqueRecsReducer extends Reducer<Text, Text, Text, Text> {
    @Override
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        HashSet<String> uniqueValues = new HashSet<>();
        for (Text val : values) {
            uniqueValues.add(val.toString());
        }
        context.write(key, new Text("ListOfUniqueValues: " + uniqueValues + " Count: " + uniqueValues.size()));
    }
}
