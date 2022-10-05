public class Percolation {

    private static final boolean OPEN = true;
    private static final boolean BLOCKED = false;

    private boolean[] grid;
    private int N, TOP, BOTTOM;
    private WeightedQuickUnionUF uf;

    // create N-by-N grid, with all sites blocked
    public Percolation(int n) {
        N = n;
        TOP = N * N;
        BOTTOM = N * N + 1;
        grid = new boolean[N * N + 2];
        for (int i = 0; i < grid.length; ++i) {
             grid[i] = BLOCKED;
        }
        uf = new WeightedQuickUnionUF(N * N + 2);
    }

    private int index(int i, int j) {
        return N * (i - 1) + (j - 1);
    }

    // open site (row i, column j) if it is not already
    public void open(int i, int j) {
        if (!isOpen(i, j)) {
            if (i > 1 && isOpen(i - 1, j)) {
                uf.union(index(i, j), index(i - 1, j));
            }
            if (i < N && isOpen(i + 1, j)) {
                uf.union(index(i, j), index(i + 1, j));
            }
            if (j > 1 && isOpen(i, j - 1)) {
                uf.union(index(i, j), index(i, j - 1));
            }
            if (j < N && isOpen(i, j + 1)) {
                uf.union(index(i, j), index(i, j + 1));
            }
            grid[index(i, j)] = OPEN;
            if (i == 1) {
                uf.union(TOP, index(i, j));
                grid[TOP] = OPEN;
            }
            if (i == N) {
                uf.union(BOTTOM, index(i, j));
                grid[BOTTOM] = OPEN;
            }
        }
    }

    public boolean isOpen(int i, int j) { // is site (row i, column j) open?
        if (i <= 0 || i > N) {
            throw new IndexOutOfBoundsException("row index " + i + " out of bounds");
        }
        if (j <= 0 || j > N) {
            throw new IndexOutOfBoundsException("row index " + j + " out of bounds");
        }
        return grid[index(i, j)];
    }

    public boolean isFull(int i, int j) { // is site (row i, column j) full?
        if (i <= 0 || i > N) {
            throw new IndexOutOfBoundsException("row index " + i + " out of bounds");
        }
        if (j <= 0 || j > N) {
            throw new IndexOutOfBoundsException("row index " + j + " out of bounds");
        }
        return uf.connected(TOP, index(i, j));
    }

    public boolean percolates() { // does the system percolate?
        return uf.connected(TOP, BOTTOM);
    }
    
    public static void main(String[] args) {
        Percolation p = new Percolation(1);
        p.open(1, 1); 
        System.out.println(p.isFull(1, 1));
        System.out.println(p.percolates());
    }
}