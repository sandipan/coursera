/*************************************************************************
 * Name: Sandipan Dey
 * Email: sandipan.dey@gmail.com
 *
 *************************************************************************/

import java.util.LinkedList;
import java.util.List;

public class Board {
	
	private int [] board;
    private int n;
    private int numMoves;
    
    // construct a board from an N-by-N array of blocks
    // (where blocks[i][j] = block in row i, column j)
	public Board(int[][] blocks)
    {
		n = blocks.length;
		board = new int[n * n];
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				board[n * i + j] 
				      = blocks[i][j] == 0 ? Integer.MAX_VALUE : blocks[i][j];
			}
		}
    }
    
	private Board(int[] board, int move, int n)
	{
		this.n = n;
		this.board = board;
		this.numMoves = move;
	}
    
	// board dimension N
	public int dimension()                 
    {
    	return n;
    }
    
	// number of blocks out of place
	public int hamming()                   
    {
    	int wrongCount = 0;
    	for (int i = 0; i < n * n; ++i) {
    	    if (board[i] == Integer.MAX_VALUE) continue;
    	    else if (board[i] != i + 1) {
    			++wrongCount;
    		}
    	}
    	return numMoves + wrongCount;
    }
    
	// sum of Manhattan distances between blocks and goal
	public int manhattan()                 
    {
    	int distance = 0;
    	for (int i = 0; i < n * n; ++i) {
    		if (board[i] == Integer.MAX_VALUE) continue;
    		int g_i = board[i] - 1;
    		distance += Math.abs(g_i / n - i / n) + Math.abs(g_i % n - i % n);
    	}
    	return numMoves + distance;
    }
    
	// is this board the goal board?
	public boolean isGoal()                
    {
    	for (int i = 0; i < n * n - 1; ++i) {
    	    if (board[i] != i + 1) {
                return false;
        	}
        }	
    	return true;
    }
    
	// a board obtained by exchanging two adjacent blocks in the same row
	public Board twin()                    
    {
		Board b = null;
		for (int i = 0; i < n * n; ++i) {
			int row = i / n;
    		int col = i % n;
    		if (col <= n - 2) {
    			if (!isSpace(row, col) && !isSpace(row, col + 1)) {
    				b = cloneBoard(this.numMoves);
    				b.swap(i, i + 1);  
    				break;
    			}
    		}    		
		}
		return b;
    }
    
	// does this board equal y?
	public boolean equals(Object y)            
    {
	    if (y == this) return true;
        if (y == null) return false;
        if (y.getClass() != this.getClass()) return false;
        Board yBoard = (Board)y;
    	int n_y = yBoard.dimension();
    	if (n != n_y) {
    		return false;
    	}
    	for (int i = 0; i < n * n; ++i) {
        	if (board[i] != yBoard.board[i]) {
        			return false;
        	}
    	}
    	return true;
    }
    
    private Board swap(int i, int j)
    {
    	int t = board[i];
    	board[i] = board[j];
    	board[j] = t;
    	return this;
    }
    
    private Board cloneBoard(int move)
    {
		int [] board = new int[n * n];
    	System.arraycopy(this.board, 0, board, 0, n * n);
		return new Board(board, move, n);
    }
    
    private boolean isSpace(int i, int j)
    {
    	return board[n * i + j] == Integer.MAX_VALUE;
    }
    
    // all neighboring boards
    public Iterable<Board> neighbors()     
    {
    	int row, col;
    	List<Board> neighbors = new LinkedList<Board>();
    	for (int i = 0; i < n * n; ++i) {
    		row = i / n;
    		col = i % n;
    		if (isSpace(row, col)) continue;
    		if (col >= 1 && isSpace(row, col - 1)) { // to left
    			neighbors.add(cloneBoard(numMoves + 1).swap(i, n * row + (col - 1)));
    		}	
    		if (col <= n - 2 && isSpace(row, col + 1)) { // to right
    			neighbors.add(cloneBoard(numMoves + 1).swap(i, n * row + (col + 1)));
    		}
    		if (row >= 1 && isSpace(row - 1, col)) { // to up
    			neighbors.add(cloneBoard(numMoves + 1).swap(i, n * (row - 1) + col));
    		}
    		if (row <= n - 2 && isSpace(row + 1, col)) { // to down
    			neighbors.add(cloneBoard(numMoves + 1).swap(i, n * (row + 1) + col));
    		}
    	}
    	return neighbors;
    }
    
    // string representation of the board (in the output format specified below)
    public String toString()               
    {
        StringBuilder s = new StringBuilder();
        s.append("\n" + n + "\n");
    	for (int i = 0; i < n * n; ++i) {
       		if (i != 0 && i % n == 0) {
       		    s.append("\n");
       		}
       		s.append(String.format("%2d ", board[i] == Integer.MAX_VALUE ? 0 : board[i]));
    	}
        return s.toString();
   }
}