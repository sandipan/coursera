from itertools import product

# takes a directed graph G with edge costs specified by C and a
# starting vertex s
# i) if there is no negative cost cycle reachable from s then it
# returns the distance labels and predecessor information
# ii) else it returns None since distance labels don't make sense and
# the list of vertices in the negative cycle
def bellmanford(G, C, s):
    d = dict()
    p = dict()
    for u in G:
        d[u] = None
        p[u] = None
    d[s] = 0

    n = len(G.keys())
    for i in range(n - 1):
        relaxed = False
        for u in G:
            if d[u] is None:
                continue
            for v in G[u]:
                if d[v] is None or d[u] + C[(u, v)] < d[v]:
                    d[v] = d[u] + C[(u, v)]
                    p[v] = u
                    relaxed = True
        #for u in G:
        #    print u, d[u]
        #print
        if not relaxed:
            break

    for u in G:
        if d[u] is None:
            continue
        for v in G[u]:
            # detecting a negative cycle
            if d[u] + C[(u, v)] < d[v]:
                d[v] = d[u] + C[(u, v)]
                p[v] = u

                # finding the negative cycle
                hare = p[p[v]]
                tortoise = p[v]
                while hare != tortoise:
                    hare = p[p[hare]]
                    tortoise = p[tortoise]

                stack = [hare]
                head = stack[-1]
                while p[head] != hare:
                    stack.append(p[head])
                    head = stack[-1]

                stack.reverse()

                return None, stack

    return d, p

if __name__ == '__main__':

    G = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['E'],
         'E': ['F'],
         'F': ['C']}
    C = {('B', 'C'): -8,
         ('E', 'F'): 5,
         ('A', 'C'): -9,
         ('D', 'E'): -3,
         ('A', 'B'): -7,
         ('C', 'D'): 4,
         ('F', 'C'): -3,
         ('B', 'D'): -10}
    s = 'A'

    G = {0: [7, 8, 13, 27, 25, 4, 29, 17, 3, 10, 20], 1: [5, 20, 11, 29, 3, 24, 10, 18, 19, 17, 2], 2: [0, 29, 24, 25, 5, 16, 22, 4, 1, 3], 3: [7, 27, 0, 12, 4, 8, 15, 5, 14, 20, 28, 24], 4: [6, 28, 23, 20, 3, 1, 12, 15, 16, 26, 25, 10, 0], 5: [29, 12, 20, 6, 16, 7, 14, 27, 11, 15, 3, 26, 17, 21, 9], 6: [1, 3, 25, 9, 28, 29, 10, 2, 11, 21, 12, 23, 19, 22, 27, 0, 14], 7: [8, 6, 17, 24, 4, 11, 29, 13, 14, 12, 21, 3, 1], 8: [6, 20, 23, 19, 5, 25, 28, 1, 4, 13, 17], 9: [29, 19, 13, 27, 6, 0, 2, 5, 24, 17], 10: [9, 28, 13, 15, 18, 0, 17, 14, 25, 5, 4, 12, 11, 7], 11: [7, 12, 8, 2, 15, 25, 20, 23, 10, 6, 5], 12: [14, 15, 17, 0, 9, 28, 24, 19, 13, 20, 29, 18], 13: [19, 3, 4, 5, 23, 27, 14, 12, 15, 24, 2], 14: [5, 15, 20, 9, 25, 27, 28, 21], 15: [4, 17, 6, 14, 11, 25, 12, 28,18, 3, 1], 16: [15, 4, 11, 1, 26, 12, 25, 3, 24, 17, 22], 17: [27, 7, 14, 20, 29, 18, 8, 5, 25, 12, 16, 21, 2, 4], 18: [7, 12, 5,28, 17, 25, 26, 21, 16, 14], 19: [9, 3, 0, 10, 22, 18, 24, 6, 2, 14, 25, 5, 28, 15, 11], 20: [12, 4, 14, 24, 8, 27, 18, 15, 25], 21: [5, 10, 16, 29, 13, 25, 18, 26, 1, 6, 28, 15, 14, 20], 22: [14, 9, 25, 13, 11, 23, 28], 23: [0, 12, 28, 11, 8, 3, 29, 14, 26,9, 22, 15, 1], 24: [1, 20, 17, 0, 28, 27, 16, 10, 12, 11, 21, 19, 29, 8, 7], 25: [26, 14, 11, 12, 29, 8, 22, 18, 6, 15, 4, 3], 26: [10, 15, 8, 11, 29, 12, 24, 23, 9, 27, 13, 5], 27: [22, 8, 14, 12, 9, 24, 2], 28: [15, 13, 8, 0, 2, 16, 22, 27, 11, 20,  26], 29: [23, 12, 22, 16, 15, 2, 17, 14, 19, 1, 6, 10, 11, 13, 20]}
    C = {(7, 3): 236, (20, 25): 218, (6, 28): 765, (21, 28): 447, (17, 20): 454, (23, 26): 234, (21, 6): 19, (8, 5): 759, (9, 0): 814, (10, 7): 406, (0, 17): 456, (12, 17): 878, (25, 15): 633, (15, 4): 445, (26, 12): 758, (29, 17): 865, (29, 11): 143, (6, 23): 640, (20, 14): 235, (21, 15): 802, (23, 9): 548, (10, 14): 805, (11, 15): 563, (9, 19): 435, (24, 21): 841, (26, 23): 944, (4, 12): 831, (28, 1): 461, (6, 14): 858, (19, 18): 453, (18, 5): 72, (10, 9): 257, (8, 25): 328, (24, 28): 120, (12, 29): 901, (25, 3): 64, (6, 1): 597, (7, 4): 920, (5, 20): 812, (6, 27): 493, (9, 29): 178, (12, 20): 611, (25, 4): 662, (29, 22): 183, (28, 13): 319, (5, 29): 248, (20, 27): 255, (16, 11): 522, (21, 26): 537, (19, 6): 913, (18, 17): 743, (22, 11): -30, (9, 6): 819, (10, 5): 83, (11, 8): 361, (24, 16): 102, (12, 19): 371, (1, 18): 516, (15, 6): 353, (2, 25): 259, (26, 10): 23, (3, 4): 191, (5, 6): 609, (6, 21): 244, (19, 15): 359, (20, 8): 336, (16, 24): 282, (21, 13): -18, (23, 11): -33, (10, 12): 35, (8, 20): 132, (9, 17): 168, (25, 22): 367, (2, 16): 628, (27, 24): 350, (5, 15): 691, (29, 2): 677, (6, 12): 146, (4, 20): 253, (17, 4): 38, (23, 0): 937, (11, 6): 870, (10, 17): 200, (13, 4): 976, (0, 7): 371, (28, 26): 603, (2, 5): 867, (3, 24): 661, (7, 6): -10, (16, 22): 73, (6, 25): 188, (19, 3): 746, (22, 28): 330, (20, 4): 451, (24, 11): 213, (27, 22): 969, (13, 23): 970, (29, 20): 522, (27, 12): 13, (4, 26): 741, (28, 15): 956, (5, 27): 463, (29, 14): 779, (17, 8): 119, (7, 21): 797, (22, 23): 97, (23, 22): 840, (22, 9): 175, (23, 12): 633, (11, 10): 893, (14, 5): 734, (12, 13): 411, (26, 8): 33, (1, 10): 104, (4, 1): -7, (16, 4): 115, (6, 11): 318, (16, 26): 161, (17, 27): 564, (9, 13): 377, (11, 25): 407, (3, 15): 755, (1, 3): 745, (6, 2): 626, (19, 22): 94, (7, 11): 188, (16, 17): 283, (21, 20): 611, (9, 24): 179, (25, 29): 25, (13, 2): 391, (0, 25): 710, (28, 20): 800, (2, 3): 382, (27, 9): 415, (5, 16): 628, (20, 24): 540, (21, 29): 176, (19, 5): 530, (17, 21): 806, (8, 4): 385, (11, 23): 541, (1, 29): 177, (25, 8): 636, (26, 15): 347, (27, 14): 746, (29, 12): 147, (16, 15): 309, (6, 22): 624, (19, 10): 661, (17, 14): 188, (23, 14): 845, (8, 17): 228, (11, 12): 252, (24, 20): 358, (12, 15): 475, (13, 14): 963, (0, 13): 255, (3, 8): 753, (4, 3): 124, (28, 0): 205, (6, 9): 678, (17, 7): 167, (7, 12): 612, (17, 25): 38, (11, 5): 466, (24, 27): 651, (10, 18): 51, (25, 26): 544, (15, 11): 178, (0, 4): 17, (24, 1): 60, (12, 28): 161, (4, 10): 357, (5, 11): 66, (6, 0): 710, (4, 16): 633, (19, 24): 864, (5, 21): 449, (21, 18): 899, (17, 18): 759, (22, 25): -25, (18, 25): 896, (23, 28): 733, (14, 21): 578, (26, 24): 552, (0, 27): 644, (24, 8): 492, (28, 22): 325, (2, 1): 357, (29, 23): 421, (3, 28): 748, (6, 29): 809, (7, 24): 840, (18, 16): 634, (21, 5): 392, (8, 6): -18, (14, 28): 69, (10, 4): -42, (15, 25): 755, (12, 18): 721, (1, 19): 228, (25, 14): 789, (13, 19): 540, (2, 24): -44, (26, 13): 22, (0, 8): 485, (29, 16): 298, (3, 5): 725, (4, 6): 210, (28, 11): -38, (5, 7): 976, (29, 10): 185, (16, 1): 799, (17, 12): 478, (7, 17): 546, (20, 15): 872, (21, 14): 960, (8, 13): 145, (22, 13): 994, (23, 8): 556, (10, 15): 337, (8, 19): 423, (14, 9): 157, (12, 9): 567, (1, 20): 229, (15, 12): -46, (13, 12): 541, (28, 2): 738, (17, 5): 960, (7, 14): -38, (18, 28): 827, (23, 1): 665, (11, 7): 72, (9, 27): 490, (24, 29): 665, (12, 0): 126, (13, 5): 73, (2, 4): 217, (5, 9): 14, (21, 16): 843, (19, 0): 456, (17, 16): 682, (7, 29): 506, (8, 1): 146, (14, 27): 518, (15, 18): 92, (0, 29): 579, (24, 10): 722, (1, 24): 70, (25, 11): 848, (13, 24): 739, (28, 16): 301, (4, 25): 457, (29, 15): 429, (16, 12): 485, (6, 19): 502, (21, 25): 548, (19, 9): 772, (18, 14): 12, (9, 5): 378, (0, 20): 219, (24, 17): 563, (1, 17): 371, (25, 12): 846, (15, 1): 581, (2, 22): 986, (26, 11): 79, (0, 10): 696, (3, 7): 286, (27, 2): 280, (1, 11): -24, (4, 0): 583, (16, 3): 798, (6, 10): 518, (19, 14): 437, (17, 2): -20, (16, 25): 157, (10, 13): 587, (14, 15): 307, (15, 14): 690, (3, 12): 124, (4, 15): 755, (5, 14): -17, (29, 1): 550, (7, 8): 57, (17, 29): 593, (18, 26): 606, (23, 3): 515, (8, 28): 446, (13, 3): 899, (26, 29): 474, (12, 24): 300, (1, 5): 278, (28, 27): 194, (3, 27): 620, (4, 28): -33, (19, 28): 295, (7, 1): 482, (5, 17): 82, (19, 2): 151, (18, 21): 389, (9, 2): 942, (14, 25): 678, (10, 25): 669, (15, 28): 719, (11, 20): 719, (24, 12): 540, (2, 29): 582, (29, 19): 540, (3, 0): 264, (28, 8): 253, (5, 26): 974, (29, 13): 327, (19, 11): 143, (20, 12): 696, (18, 12): 493, (21, 1): 976, (22, 14): 285, (23, 15): 418, (10, 0): 306, (24, 19): 244, (12, 14): 281, (25, 18): 799, (15, 3): 224, (13, 15): 480, (26, 9): 162, (5, 3): 813, (29, 6): 492, (7, 13): 616, (18, 7): 312, (21, 10): 666, (10, 11): 106, (8, 23): 9, (11, 2): 39, (0, 3): 612, (24, 0): 982, (3, 14): 718, (1, 2): 981, (5, 12): 399, (3, 20): 451, (6, 3): 450, (4, 23): 554, (19, 25): 387, (20, 18): -6, (23, 29): 611, (14, 20): 449, (10, 28): 983, (15, 17): 383, (26, 27): 157, (24, 7): 505, (25, 6): 610, (13, 27): 746, (2, 0): 722, (26, 5): 346, (27, 8): 284}
    s = 24
	
    G = {'s': ['t', 'y'],
         'y': ['z', 'x'],
         't': ['x', 'y', 'z'],
         'x': ['t'],
         'z': ['s', 'x']}
    C = {('s', 't'): 6,
         ('s', 'y'): 7,
         ('t', 'y'): 8,
         ('y', 'z'): 9,
         ('z', 's'): 2,
         ('y', 'x'): -3,
         ('z', 'x'): 7,
         ('x', 't'): -2,
		 ('t', 'x'): 5,
         ('t', 'z'): -4}
    s = 's'
	
    d, p = bellmanford(G, C, s)

    if d is None:
        print 'negative cycle detected:', p
    else:
        for u in G:
            if d[u] is None:
                print u, 'is unreachable'
            else:
                print u, ':', d[u], ':',

                stack = [u]
                head = stack[-1]
                while p[head]:
                    stack.append(p[head])
                    head = stack[-1]

                while stack:
                    print stack.pop(),
                print
				
    G = {0: [4, 5, 6, 7, 9, 11, 13, 16, 18, 21, 22, 23, 24, 26, 27], 1: [3, 4, 6, 7, 9, 10, 11, 15, 16, 18, 23, 27], 2: [0, 1, 4, 6, 7, 8, 9, 11, 14, 15, 16, 18, 24, 25, 26, 27], 3: [1, 2, 5, 8, 12, 13, 17, 18, 24, 25, 26], 4: [0, 3, 5, 7, 8, 10, 12, 14, 15, 24, 27], 5: [0, 1, 3, 7, 9, 10, 11, 14, 16, 19, 20, 24, 28, 29], 6: [4, 5, 7, 9, 10, 11, 13, 15, 17, 19, 20, 21, 26, 27], 7: [1, 2, 3, 4, 6, 9, 10, 13, 18, 21, 25, 27], 8: [3, 4, 6, 7, 10, 18, 19, 21, 22, 27, 28, 29], 9: [1, 8, 16, 20, 25, 29], 10: [4, 7, 8, 14, 15, 16, 17, 18, 23, 26, 28], 11: [1, 2, 3, 5, 7, 8, 9, 10, 13, 22, 23, 27, 29], 12: [1, 4, 5, 6, 7, 9, 10, 11, 14, 16, 18, 19, 22, 23, 27, 28], 13: [0, 1, 2, 3, 4, 9, 12, 23, 26, 28], 14: [0, 5, 6, 7, 8, 9, 10, 11, 17, 23, 27, 28], 15: [5, 8, 11, 13, 20, 22, 23, 26, 27], 16: [5, 6, 9, 10, 13, 17, 18, 19, 22, 23, 24, 25, 26], 17: [0, 1, 7, 9, 10, 14, 15, 18, 21, 24, 25, 27, 29], 18: [4, 7, 8, 9, 12, 13, 14, 19, 20, 22, 23, 26, 27], 19: [1, 3, 4, 7, 9, 14, 21, 24, 27, 29], 20: [0, 5, 7, 10, 12, 13, 17, 21, 23, 24, 26, 27, 28, 29], 21: [1, 5, 10, 12, 15, 19, 20, 22, 23, 25, 26, 27, 28, 29], 22: [2, 4, 5, 6, 9, 10, 11, 16, 21, 25, 26, 27], 23: [0, 4, 5, 6, 7, 9, 10, 11, 12, 17, 18, 21, 22, 24, 25, 28, 29], 24: [0, 3, 8, 10, 12, 13, 16, 23, 26, 27, 28], 25: [0, 2, 9, 10, 12, 18, 19, 20, 23, 26, 28, 29], 26: [1, 2, 10, 11, 14, 15, 16, 18, 23, 24, 28], 27: [5, 7, 10, 13, 18, 19, 21, 22, 25, 26, 28, 29], 28: [0, 6, 7, 9, 10, 12, 13, 15, 16, 21, 22, 25, 26, 29], 29: [0, 1, 2, 4, 7, 11, 13, 17, 22, 23, 24, 25, 26, 27, 28]}
    C = {(7, 3): 558, (16, 9): 913, (21, 28): 902, (19, 4): 144, (7, 25): 706, (20, 7): 175, (18, 19): 767, (10, 7): 637, (11, 22): 55, (2, 27): 106, (29, 17): 736, (3, 2): 648, (4, 5): 783, (28, 10): 225, (5, 24): 69, (29, 11): 78, (21, 15): 91, (23, 9): 398, (10, 14): 388, (8, 18): 386, (14, 8): 80, (15, 13): 248, (2, 18): 373, (26, 23): 781, (1, 15): 79, (4, 12): 66, (5, 1): 382, (29, 4): 795, (3, 17): 61, (20, 21): 734, (17, 24): 796, (23, 6): 707, (9, 20): 29, (24, 28): 236, (12, 7): 779, (0, 5): 388, (2, 7): 32, (5, 10): 853, (19, 27): 133, (7, 4): 586, (5, 20): 543, (20, 28): 543, (6, 27): 423, (19, 1): 118, (18, 22): 853, (9, 29): 421, (10, 26): -41, (29, 22): 547, (27, 10): 729, (4, 24): 350, (28, 13): 668, (5, 29): 417, (20, 27): 275, (21, 26): 383, (17, 10): 596, (7, 27): 384, (8, 7): 889, (22, 11): 248, (11, 8): 946, (24, 16): 693, (14, 7): 908, (12, 19): 658, (1, 18): 469, (2, 25): 791, (26, 10): 201, (0, 9): 861, (4, 7): 38, (6, 21): 317, (18, 8): -30, (16, 24): 843, (22, 2): 573, (23, 11): 353, (11, 1): 734, (24, 23): 229, (12, 10): 697, (2, 16): 697, (3, 13): 177, (4, 14): 958, (29, 2): 651, (7, 9): 554, (20, 23): 229, (21, 22): 306, (22, 5): 84, (23, 0): 600, (8, 27): 74, (14, 17): 2, (12, 1): 227, (10, 17): -14, (15, 20): 53, (13, 4): 458, (26, 28): 75, (0, 7): 35, (1, 6): -7, (28, 26): 323, (29, 27): 441, (3, 24): -5, (6, 7): 731, (19, 29): 753, (7, 6): 932, (16, 22): 873, (19, 3): 863, (18, 20): 208, (23, 25): 237, (12, 22): 201, (27, 22): 783, (25, 10): 108, (13, 23): -29, (26, 1): 491, (3, 1): 524, (28, 15): 680, (16, 13): 54, (7, 21): 685, (23, 22): 288, (22, 9): 37, (23, 12): 160, (15, 26): 869, (11, 10): 950, (0, 21): 444, (14, 5): 875, (1, 16): 929, (25, 19): 426, (0, 11): 465, (27, 5): 883, (1, 10): 14, (28, 6): 909, (29, 7): 801, (6, 11): 144, (17, 1): 488, (7, 18): 670, (20, 10): 438, (16, 26): 760, (17, 27): 478, (23, 5): 997, (8, 22): 589, (11, 3): 582, (12, 4): 391, (25, 20): 212, (13, 9): 158, (27, 26): 723, (1, 3): 683, (4, 8): 726, (28, 29): 749, (2, 8): 18, (29, 0): 676, (20, 17): 599, (16, 17): 500, (21, 20): 343, (22, 27): 871, (18, 27): 363, (8, 29): 89, (14, 23): 79, (25, 29): 513, (15, 22): 323, (13, 2): 300, (27, 19): 503, (1, 4): 440, (13, 28): 993, (29, 25): 650, (3, 26): 920, (6, 5): 241, (5, 16): 297, (20, 24): 739, (21, 29): 194, (17, 21): 455, (8, 4): 787, (9, 1): 666, (11, 23): 77, (0, 16): 927, (24, 13): 334, (14, 0): 317, (12, 16): 955, (15, 5): 334, (2, 26): 748, (26, 15): 643, (28, 9): 333, (17, 14): 863, (22, 21): 359, (20, 13): 918, (18, 13): 1000, (0, 23): 688, (14, 11): 345, (0, 13): 401, (3, 8): 167, (27, 7): 323, (4, 3): 208, (28, 0): 470, (2, 15): 710, (16, 6): 393, (6, 9): 997, (17, 7): 223, (18, 4): 101, (17, 25): 854, (22, 6): 619, (23, 7): 488, (10, 8): 391, (11, 5): 380, (24, 27): 706, (12, 6): 86, (10, 18): 959, (25, 26): 168, (15, 11): 189, (11, 27): 946, (0, 4): 997, (12, 28): 120, (27, 28): 341, (4, 10): 177, (2, 6): 166, (5, 11): 118, (19, 24): 941, (16, 19): 695, (6, 26): 500, (17, 18): 572, (22, 25): 100, (23, 28): 175, (13, 0): 658, (26, 24): 144, (0, 27): 22, (24, 8): 445, (12, 27): 710, (27, 21): -7, (13, 26): 697, (28, 22): 82, (2, 1): 291, (26, 2): 997, (29, 23): 571, (28, 12): 60, (7, 2): 362, (20, 26): 105, (16, 10): 673, (21, 27): 463, (19, 7): 585, (22, 16): 158, (20, 0): 777, (23, 21): 495, (21, 5): 957, (8, 6): 566, (22, 10): 879, (14, 28): 48, (10, 4): 518, (11, 9): 516, (0, 18): 787, (14, 6): 879, (12, 18): 138, (2, 24): 585, (3, 5): 383, (5, 7): 661, (6, 20): 831, (23, 18): 784, (9, 8): 655, (10, 15): 254, (8, 19): 391, (14, 9): 862, (12, 9): 764, (25, 23): 685, (13, 12): 988, (27, 25): 435, (5, 0): 145, (6, 15): 843, (19, 21): -7, (21, 23): 583, (22, 4): 203, (11, 7): 457, (10, 16): 119, (11, 29): 520, (0, 6): 769, (24, 3): 430, (1, 7): 11, (25, 2): 68, (28, 25): 168, (2, 4): 183, (5, 9): 246, (29, 28): 737, (3, 25): 476, (5, 19): 455, (20, 29): 209, (18, 23): 168, (14, 27): 396, (24, 10): 480, (28, 16): 898, (27, 13): 816, (5, 28): 640, (6, 19): 287, (21, 25): -17, (19, 9): 451, (17, 9): 742, (18, 14): 223, (15, 27): 100, (25, 12): 579, (26, 11): 123, (1, 11): 314, (4, 0): 204, (6, 10): -36, (19, 14): 854, (18, 9): 641, (16, 25): 41, (21, 12): 870, (23, 10): 324, (8, 21): 123, (9, 16): 250, (12, 11): 28, (10, 23): 916, (26, 18): 53, (3, 12): 15, (4, 15): 459, (2, 11): 33, (5, 14): 859, (29, 1): -39, (3, 18): -16, (6, 13): 587, (17, 29): 369, (22, 26): 797, (18, 26): 173, (8, 28): 360, (9, 25): 859, (15, 23): 280, (13, 3): 12, (0, 24): 898, (25, 0): 942, (29, 26): 882, (6, 4): 307, (7, 1): 435, (16, 23): 667, (20, 5): 320, (23, 24): 287, (8, 3): 643, (24, 12): 679, (12, 23): 358, (25, 9): 437, (26, 14): 918, (4, 27): 74, (29, 13): 808, (6, 17): 647, (17, 15): 620, (20, 12): 735, (18, 12): 683, (23, 17): 342, (21, 1): 104, (8, 10): 297, (11, 13): 352, (0, 22): -17, (14, 10): 910, (12, 14): 458, (1, 23): -18, (25, 18): -13, (1, 9): 44, (28, 7): 20, (2, 14): -46, (5, 3): 25, (16, 5): 733, (17, 0): 854, (7, 13): 226, (18, 7): 770, (21, 10): 775, (23, 4): 29, (11, 2): 399, (24, 26): 762, (12, 5): 922, (15, 8): 709, (26, 16): 159, (24, 0): 953, (27, 29): 689, (2, 9): 539, (7, 10): 156, (16, 18): -7, (21, 19): 883, (23, 29): 960, (10, 28): 518, (25, 28): 744, (13, 1): 908, (0, 26): 810, (27, 18): 123, (1, 27): 453, (28, 21): 461, (2, 0): 449, (29, 24): 441}
    s = 25    