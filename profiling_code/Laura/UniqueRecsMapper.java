import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class UniqueRecsMapper extends Mapper<LongWritable, Text, Text, Text> {
    private boolean isFirstRow = true;

    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // Skip the first row of header
        if (isFirstRow) {
            isFirstRow = false;
            return;
        }

        String[] line = value.toString().split(",");

        // pick "ZipCode", "State", and "ShipMode" columns to explore
        if (line.length > 5) {
            context.write(new Text("ZipCode"), new Text(line[line.length - 1].trim()));
            context.write(new Text("State"), new Text(line[line.length - 3].trim()));
            context.write(new Text("ShipMode"), new Text(line[line.length - 5].trim()));
        }
    }
}
