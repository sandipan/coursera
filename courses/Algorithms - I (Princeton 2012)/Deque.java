import java.util.Iterator;


public class Deque<Item> implements Iterable<Item> {

    private Node<Item> first, last;
    private int size;

    @SuppressWarnings("hiding")
    private class Node<Item> {
        private Item item;
        private Node<Item> left, right;
    }

    public Deque()                     // construct an empty deque
    {
        first = null;
        last = null;
        size = 0;
    }

    public boolean isEmpty()           // is the deque empty?
    {
        return size == 0;
    }

    public int size()                  // return the number of items on the deque
    {
        return size;
    }

    public void addFirst(Item item)    // insert the item at the front
    {
        if (item == null) {
            throw new java.lang.NullPointerException("Can't add Null element!");
        }
        Node<Item> cur = new Node<Item>();
        cur.item = item;
        cur.right = first;
        cur.left = null;
        if (first != null) {
            first.left = cur;
        }
        first = cur;
        if (last == null) {
            last = first;
        }
        ++size;
    }

    public void addLast(Item item)     // insert the item at the end
    {
        if (item == null) {
            throw new java.lang.NullPointerException("Can't add Null element!");
        }
        Node<Item> cur = new Node<Item>();
        cur.item = item;
        cur.left = last;
        cur.right = null;
        if (last != null) {
            last.right = cur;
        }
        last = cur;
        if (first == null) {
            first = last;
        }
        ++size;
    }

    public Item removeFirst()          // delete and return the item at the front
    {
        if (size == 0) {
            throw new java.util.NoSuchElementException("Can't remove, Empty queue!");
        }
        Node<Item> cur = first;
        first = first.right;
        if (first != null) {
            first.left = null;
        }
        if (size == 1) {
            last = first;
        }
        --size;
        return cur.item;
    }

    public Item removeLast()           // delete and return the item at the end
    {
        if (size == 0) {
            throw new java.util.NoSuchElementException("Can't remove, Empty queue!");
        }
        Node<Item> cur = last;
        last = last.left;
        if (last != null) {
            last.right = null;
        }
        if (size == 1) {
            first = last;
        }
        --size;
        return cur.item;
    }

    public Iterator<Item> iterator()
    {
        return new DequeIterator<Item>(first);
    }

    @SuppressWarnings({ "rawtypes", "hiding" })
    private class DequeIterator<Item> implements Iterator 
    {
        private Node<Item> current;

        public DequeIterator(Node<Item> first) {
            current = first;
        }

        public void remove() {
            throw new java.lang.UnsupportedOperationException("Can't remove!");
        }

        public Item next() {
            if (current == null) {
                throw new 
                java.util.NoSuchElementException("Can't return, Empty queue!");
            }
            Item item = current.item;
            current = current.right;
            return item;
        }

        public boolean hasNext() {
            return current != null;
        }
    }

    public static void main(String[] args)
    {
        Deque<Integer> d = new Deque<Integer>();
        for (int i = 1; i <= 5; ++i) {
            d.addFirst(i);
        }
        for (int i = 6; i <= 10; ++i) {
            d.addLast(i);
        }

        Iterator<Integer> iterator = d.iterator();
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }

        System.out.println(d.removeFirst());
        System.out.println(d.removeLast());

        iterator = d.iterator();
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
        for (int i = 1; i <= 10; ++i) {
            System.out.println(d.removeFirst());
        }   
    }
}