import java.io.*;
import java.util.*;

public class Harness {
    public static void main(String[] args) throws IOException {
        int w = Integer.parseInt(args[0]);
        VariableCounter[] c = new VariableCounter[w];
        for (int i = 0; i < w; i++) c[i] = new VariableCounter();
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = br.readLine()) != null) {
            String[] p = line.trim().split(" ");
            int reps = p.length > 3 ? Integer.parseInt(p[3]) : 1;
            for (int r = 0; r < reps; r++)
                c[Integer.parseInt(p[0])].increaseValue(Integer.parseInt(p[2]), Integer.parseInt(p[1]));
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < w; i++)
            sb.append(i).append(' ').append(c[i].getI()).append(' ').append(c[i].getValue(0)).append(' ').append(c[i].getValue(1)).append('\n');
        System.out.print(sb);
    }
}
