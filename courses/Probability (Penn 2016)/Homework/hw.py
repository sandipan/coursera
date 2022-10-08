# hw5
from math import factorial

def prob(n, p, k):
	return (p)**k * (1-p)**(n-k) * factorial(n) / (factorial(k) * factorial(n-k))
	#return (p)**k * (1-p)**(n-k) * factorial(n) // factorial(k) // factorial(n-k) 
	
#def prob_ge(n, p, k):
#	return sum([prob(n, p, x) for x in range(k, n + 1)])
	
def exp_profit(n, p, purchase, profit, loss):
	return sum([(min(purchase, demand) * profit - max(purchase - demand, 0) * loss) * prob(n, p, demand) for demand in range(n+1)]) #- max(purchase - n, 0) * loss
	#return purchase * profit * prob_ge(n, p, purchase) - sum([(purchase - demand) * loss * prob(n, p, purchase) for demand in range(purchase)])
	
for k in range(10):
	print k, exp_profit(5, 2.0 / 3, k, 5, 5)
	
#for i in range(6):
#	print prob(5, 2.0 / 3, i)


# hw6
# toss a coin
from random import random
r = random(); print 'Head' if r >= 0.5 else 'Tail'



# hw7
#Q1
import itertools
#itertools.permutations([1,2,3,4,5,6])
cups = ['R', 'R', 'W', 'W', 'B', 'B']
sauccers = ['R', 'R', 'W', 'W', 'B', 'B']
count, total = 0, 0
for perm in list(itertools.permutations(range(6))):
	found = True
	print perm
	for i in range(6):
		if cups[perm[i]] == sauccers[i]:
			found = False
			break	
	if found:
		count += 1
	total += 1

print count, total, (count * 1.0) / total

#Q3. R
2 * (1 - pnorm(0.2 / (sqrt(0.7 * (1 - 0.7) / 10))))














































































































































































































































