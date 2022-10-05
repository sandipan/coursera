/*************************************************************************
 * Name: Sandipan Dey
 * Email: sandipan.dey@gmail.com
 *
 * Compilation:  javac Fast.java
 * Execution:
 * Dependencies: StdDraw.java
 *
 * Description: .
 *
 *************************************************************************/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.PriorityQueue;
import java.util.Random;

public class Fast {

    private LinkedList<Point> points = new LinkedList<Point>();
    // (slope,intercept) pair as key
    private HashMap<String, String> lines = new HashMap<String, String>();
    Random r = new Random();
    
    private void findCollinearPoints() {
        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);
        while (points.size() > 0) {
            PriorityQueue<Point> colP = new PriorityQueue<Point>(points.size(),
                    new Comparator<Point>() {
                public int compare(Point p1, Point p2) {
                    if (p1.slopeTo(p2) == Double.POSITIVE_INFINITY
                            || p1.slopeTo(p2) == 0) {
                        return p1.compareTo(p2);
                    }
                    if (p1.slopeTo(p2) * p1.compareTo(p2) < 0) {
                        return -1;
                    }
                    return 1;
                }
            });
            Point p = points.removeFirst();
            Collections.sort(points, p.SLOPE_ORDER);
            ListIterator<Point> itr = points.listIterator();
            double slope = Double.NEGATIVE_INFINITY;
            double curSlope = Double.NEGATIVE_INFINITY;
            Point cur = null;
            if (itr.hasNext()) {
                cur = itr.next();
            }
            while (itr.hasNext()) {
                slope = p.slopeTo(cur);
                colP.add(cur);
                while (itr.hasNext()) {
                    cur = itr.next();
                    curSlope = p.slopeTo(cur);
                    if (curSlope != slope) {
                        break;
                    }
                    colP.add(cur);
                }
                if (colP.size() >= 3) {
                    colP.add(p);
                    Point first = colP.peek();
                    if (slope == -0.0) {
                        slope = 0.0;
                    }
                    String[] xy = first.toString().split("[\\(,\\s\\)]+");
                    int x = Integer.parseInt(xy[1]);
                    int y = Integer.parseInt(xy[2]);
                    double intercept = y - slope * x;
                    if (slope == Double.POSITIVE_INFINITY || slope == Double.NEGATIVE_INFINITY) {
                        intercept = x; 
                    }
                    String lineKey = String.valueOf(slope) + "," + intercept;
                    String existingLine = lines.get(lineKey);
                    Point last = first;
                    String newLine = "";
                    while (true) {
                        last = colP.poll();
                        newLine += last;
                        if (colP.size() == 0) {
                            break;
                        }
                        newLine += "->";
                    }
                    if (existingLine == null || 
                        newLine.split("->").length > existingLine.split("->").length) {
                        lines.put(lineKey, newLine);
                        System.out.println(newLine + ": " + existingLine);
                    }
                } 
                colP.clear();
            }
        }
        for (String line:lines.values()) {
            String [] points = line.split("->");
            Point first = null;
            Point last = null;
            for (int i = 0; i < points.length; ++i) {
                String[] pxy = points[i].split("[\\(,\\s\\)]+");
                Point p = new Point(Integer.parseInt(pxy[1]), Integer.parseInt(pxy[2]));
                p.draw();
                if (i == 0) {
                    first = p;
                }
                else if (i == points.length - 1) {
                    last = p;
                }
            }
            first.drawTo(last);
        }
    }

    public static void main(String[] args) {

        if (args.length != 1) {
            System.out.println("<usage>: java Fast <input_file>");
            return;
        }
        Fast fast = new Fast();
        HashSet<String> set = new HashSet<String>();
        String p = null;
        for (int i = 0; i < 200; ++i) {
            int x, y;
            do {
                x = fast.r.nextInt(20);
                y = fast.r.nextInt(20);
                p = x + ", " + y;
            }
            while (set.contains(p));
            fast.points.add(new Point(x, y));
            set.add(p);
        }
        fast.findCollinearPoints();
        /*
        try {
            BufferedReader br = new BufferedReader(new FileReader(args[0]));
            int n = Integer.parseInt(br.readLine());
            int i = 1;
            while (i <= n) {
                try {
                    String[] coords = br.readLine().trim().split("\\s+");
                    fast.points.add(new Point(Integer.parseInt(coords[0]),
                            Integer.parseInt(coords[1])));
                    ++i;
                } catch (NumberFormatException e) {
                    // System.out.println();
                }
            }
            fast.findCollinearPoints();
        } catch (IOException e) {
            System.err.println(e);
        }*/
    }
}