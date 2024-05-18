import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;

public class CleanMapper extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] columns = line.split(",");
        // if any column contains '\N'
        for(String column : columns) {
            if(column.trim().equals("\\N")) {
                return; // Skip
            }
        }

        StringBuilder cleanedLine = new StringBuilder();
        for (int i = 0; i < columns.length; i++) {
            // Skip the columns at indices 2, 7, and 14
            if (i != 2 && i != 7 && i != 14) {
                cleanedLine.append(columns[i]).append(",");
            }
        }

        if (cleanedLine.length() > 0) {
            cleanedLine.setLength(cleanedLine.length() - 1);
        }

        context.write(new Text(cleanedLine.toString()), new Text(""));
    }
}

