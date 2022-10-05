/*************************************************************************
 * Name: Sandipan Dey
 * Email: sandipan.dey@gmail.com
 *
 *************************************************************************/

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;

public class Solver {

    private LinkedList<Board> solution;
    private Comparator<Board> nodeComparator = new Comparator<Board>() {
        public int compare(Board b1, Board b2) {
            if (b1.manhattan() < b2.manhattan()) {
                return -1;
            }
            if (b1.manhattan() > b2.manhattan()) {
                return 1;
            }
            return 0;
        }
    };
    
    // find a solution to the initial board (using the A* algorithm)
    public Solver(Board initial) {
    	solve(initial);
    }
    
    private void solve(Board initial) {
    	
    	if (initial.isGoal()) {
    		solution = new LinkedList<Board>();
            solution.addFirst(initial);
            return;
    	}
    	
        HashMap<Board, Board> parents = new HashMap<Board, Board>();
        MinPQ<Board> boards = new MinPQ<Board>(nodeComparator);
        
        //boards.insert(initial);
        //parents.put(initial, null);
        for (Board b : initial.neighbors()) {
            boards.insert(b);
            parents.put(b, initial);
        }

        Board twin = initial.twin();
        //boards.insert(twin);
        //parents.put(twin, null);
        for (Board b : twin.neighbors()) {
            boards.insert(b);
            parents.put(b, twin);
        }

        Board cur = null;
        while (true) {
            cur = boards.delMin();
            if (cur.isGoal()) {
                break;
            }
            Board prevNode = parents.get(cur);
            for (Board b : cur.neighbors()) {
                if (b.equals(prevNode)) {
                    continue;
                }
                boards.insert(b);
                parents.put(b, cur);
            }
        }

        LinkedList<Board> solution = new LinkedList<Board>();
        solution.addFirst(cur);
        while (true) {
            Board parent = parents.get(cur);
            solution.addFirst(parent);
            cur = parent;
            if (cur == initial || cur == twin) {
                break;
            }
        }
        if (cur == initial) {
            this.solution = solution;
        }
    }

    // is the initial board solvable?
    public boolean isSolvable() {
        return solution != null;
    }

    // min number of moves to solve initial board; -1 if no solution
    public int moves() {
        return solution == null ? -1 : solution.size() - 1;
    }

    // sequence of boards in a shortest solution; null if no solution
    public Iterable<Board> solution() {
        return solution;
    }

    // solve a slider puzzle (given below)
    public static void main(String[] args) {
        // create initial board from file
        In in = new In(args[0]);
        int N = in.readInt();
        int[][] blocks = new int[N][N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                blocks[i][j] = in.readInt();
            }
        Board initial = new Board(blocks);

        // solve the puzzle
        Solver solver = new Solver(initial);

        // print solution to standard output
        if (!solver.isSolvable())
            StdOut.println("No solution possible");
        else {
            StdOut.println("Minimum number of moves = " + solver.moves());
            for (Board board : solver.solution())
                StdOut.println(board);
        }
    }
}