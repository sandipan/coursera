/*************************************************************************
 * Name: Sandipan Dey
 * Email: sandipan.dey@gmail.com
 *
 *************************************************************************/

import java.util.LinkedList;
import java.util.List;

public class KdTree {

    private Node root;             // root of KdTree
    private int size;		   // size of KdTree
    
    private static class Node {
        private Point2D p; // the point
        private RectHV rect; // the axis-aligned rectangle 
                             // corresponding to this node
        private Node lb; // the left/bottom subtree
        private Node rt; // the right/top subtree
        
        public Node(Point2D p, RectHV rect) {
            this.p = p;
            this.rect = rect;
        }
    }

    public KdTree() { // construct an empty set of points
        root = null;
        size = 0;
    }

    public boolean isEmpty() { // is the set empty?
        return size == 0; //root == null;
    }

    public int size() { // number of points in the set
        return size;
    }
    
    public void insert(Point2D p) { // add the point p to the set (if it is not
                                    // already in the set)
        root = insert(root, p, 0);
    }

    private Node insert(Node node, Point2D p, int level) {
        if (node == null) {
        	++size;
            return new Node(p, new RectHV(p.x(), p.y(), p.x(), p.y()));
        }
        else if (node.p.equals(p)) {
        	return node;
        }
        boolean less = true;
        if (level % 2 == 0) {
            less = p.x() < node.p.x(); 
        }
        else {
            less = p.y() < node.p.y(); 
        }
        double xmin = node.rect.xmin();
        double ymin = node.rect.ymin();
        double xmax = node.rect.xmax();
        double ymax = node.rect.ymax();
        if (less) {
            node.lb = insert(node.lb, p, level + 1);
            if (node.lb.rect.xmin() < xmin) {
                xmin = node.lb.rect.xmin();
            }
            if (node.lb.rect.ymin() < ymin) {
                ymin = node.lb.rect.ymin();
            }
            if (node.lb.rect.xmax() > xmax) {
                xmax = node.lb.rect.xmax();
            }
            if (node.lb.rect.ymax() > ymax) {
                ymax = node.lb.rect.ymax();
            }
        }
        else {
            node.rt = insert(node.rt, p, level + 1);
            if (node.rt.rect.xmin() < xmin) {
                xmin = node.rt.rect.xmin();
            }
            if (node.rt.rect.ymin() < ymin) {
                ymin = node.rt.rect.ymin();
            }
            if (node.rt.rect.xmax() > xmax) {
                xmax = node.rt.rect.xmax();
            }
            if (node.rt.rect.ymax() > ymax) {
                ymax = node.rt.rect.ymax();
            }
        }
        node.rect = new RectHV(xmin, ymin, xmax, ymax);
        return node;
    }
    
    public boolean contains(Point2D p) { // does the set contain the point p?
        return get(root, p, 0) != null;
    }

    private Node get(Node node, Point2D p, int level) {
        if (node == null) {
            return null;
        }
        int cmp = 0;
        if (level % 2 == 0) {
            if (p.x() < node.p.x()) {
               cmp = -1;
            }
            else if (p.x() > node.p.x()) {
                cmp = 1;
            }            
        }
        else {
             if (p.y() < node.p.y()) {
                cmp = -1;
             }
             else if (p.y() > node.p.y()) {
                 cmp = 1;
             }            
        }
        if (cmp < 0) {
            return get(node.lb, p, level + 1);
        }
        else if (cmp > 0) {
            return get(node.rt, p, level + 1);
        }
        else {
        	if (node.p.equals(p)) {
                return node;
        	}
        	else {
                return get(node.rt, p, level + 1);
        	}
        }
    }

    /*private void preOrder() {
        preOrder(root);
    }
    
    private void preOrder(Node node) {
        if (node != null) {
            System.out.println(node.p.x() + "," + node.p.y() + ": " +node.rect);
            preOrder(node.lb);
            preOrder(node.rt);
        }
    }*/
    
    public void draw() { // draw all of the points to standard draw
        draw(root, 0, -1, 1);
    }

    private void draw(Node node, int level, double min, double max) { // draw all of the points to standard draw
        if (node != null) {
            StdDraw.setPenColor(StdDraw.BLACK);
            StdDraw.setPenRadius(.01);
            StdDraw.point(node.p.x(), node.p.y());
            StdDraw.setPenRadius();
            if (level % 2 == 0) {
                StdDraw.setPenColor(StdDraw.RED);
                StdDraw.line(node.p.x(), min, node.p.x(), max);
            }
            else {
                StdDraw.setPenColor(StdDraw.BLUE);
                StdDraw.line(min, node.p.y(), max, node.p.y());
            }
            draw(node.lb, level + 1, min, max);
                    //level % 2 == 0 ? node.p.x() : node.p.y());
            draw(node.rt, level + 1, min, max); 
                    //level % 2 == 0 ? node.p.x() : node.p.y(), max);
        }
    }

    public Iterable<Point2D> range(RectHV rect) { // all points in the set that
                                                  // are inside the rectangle
        List<Point2D> list = new LinkedList<Point2D>();
        if (root != null) {
            range(root, list, rect);
        }
        return list;
    }

    private void range(Node node, List<Point2D> list, RectHV query) { // all points in the set that
        if (node != null && node.rect.intersects(query)) {
            if (query.contains(node.p)) {
                list.add(node.p);
            }
            range(node.lb, list, query);
            range(node.rt, list, query);
        }
    }

    public Point2D nearest(Point2D p) { // a nearest neighbor in the set to p;
                                        // null if set is empty
        if (root != null) {
            return nearest(root, p, Double.MAX_VALUE);
        }
        return null;
    }
   
    private Point2D nearest(Node node, Point2D query, double minDist) {
        Point2D p = node.p;
        double d = node.rect.distanceSquaredTo(query);
        if (d < minDist) {
            minDist = d;
            Point2D p1;
        	if (node.lb != null && node.rt != null) {
        		double d1 = node.lb.rect.distanceSquaredTo(query);
        		double d2 = node.rt.rect.distanceSquaredTo(query);
        		if (d1 < d2) {
        		    p1 = nearest(node.lb, query, minDist);
        		    if (distSqr(p1, query) < distSqr(p, query)) {
        		    	p = p1;
        		    }
        		    p1 = nearest(node.rt, query, minDist);
        		    if (distSqr(p1, query) < distSqr(p, query)) {
        		    	p = p1;
        		    }
        		}
        		else {
        		    p1 = nearest(node.rt, query, minDist);
        		    if (distSqr(p1, query) < distSqr(p, query)) {
        		    	p = p1;
        		    }
        		    p1 = nearest(node.lb, query, minDist);
        		    if (distSqr(p1, query) < distSqr(p, query)) {
        		    	p = p1;
        		    }
        		}
        	}
        	else if (node.lb != null) {
    		    p1 = nearest(node.lb, query, minDist);
    		    if (distSqr(p1, query) < distSqr(p, query)) {
    		    	p = p1;
    		    }
            }
        	else if (node.rt != null) {
    		    p1 = nearest(node.rt, query, minDist);
    		    if (distSqr(p1, query) < distSqr(p, query)) {
    		    	p = p1;
    		    }
            }
        }
        return p;
    }
    
    private double distSqr(Point2D p1, Point2D p2) {
        return (p1.x() - p2.x()) * (p1.x() - p2.x()) + 
               (p1.y() - p2.y()) * (p1.y() - p2.y());
    }
    
    public static void main(String[] args) {
       
        KdTree tree = new KdTree();
        PointSET pset = new PointSET();
        In in = new In(args[0]);
        int N = 10;
        double x, y;
        for (int i = 0; i < N; i++) {
           x = in.readDouble();
           y = in.readDouble(); 
           tree.insert(new Point2D(x, y));
           pset.insert(new Point2D(x, y));
        }    
        //tree.draw();
        //tree.preOrder();
        /*for (Point2D p:tree.range(new RectHV(0,0, 0.81, 0.3))) {
            System.out.println(p);
        }*/
        for (double i = .1; i > .00001; i = i / 10) {
            for (x = 0; x < 1; x += i) {
                for (y = 0; y < 1; y += i) {
                	System.out.println(tree.nearest(new Point2D(x, y)) + ": " + pset.nearest(new Point2D(x, y)));
                }
            }
        }
    }
}