#import numbthy

import math  # Use sqrt, floor

def gcd(a,b):
	"""gcd(a,b) returns the greatest common divisor of the integers a and b."""
	if a == 0:
		return b
	return abs(gcd(b % a, a))
	
def powmod(b,e,n):
	"""powmod(b,e,n) computes the eth power of b mod n.  
	(Actually, this is not needed, as pow(b,e,n) does the same thing for positive integers.
	This will be useful in future for non-integers or inverses."""
	accum = 1; i = 0; bpow2 = b
	while ((e>>i)>0):
		if((e>>i) & 1):
			accum = (accum*bpow2) % n
		bpow2 = (bpow2*bpow2) % n
		i+=1
	return accum
	
def xgcd(a,b):
	"""xgcd(a,b) returns a list of form [g,x,y], where g is gcd(a,b) and
	x,y satisfy the equation g = ax + by."""
	a1=1; b1=0; a2=0; b2=1; aneg=1; bneg=1; swap = False
	if(a < 0):
		a = -a; aneg=-1
	if(b < 0):
		b = -b; bneg=-1
	if(b > a):
		swap = True
		[a,b] = [b,a]
	while (1):
		quot = -(a / b)
		a = a % b
		a1 = a1 + quot*a2; b1 = b1 + quot*b2
		if(a == 0):
			if(swap):
				return [b, b2*bneg, a2*aneg]
			else:
				return [b, a2*aneg, b2*bneg]
		quot = -(b / a)
		b = b % a;
		a2 = a2 + quot*a1; b2 = b2 + quot*b1
		if(b == 0):
			if(swap):
				return [a, b1*bneg, a1*aneg]
			else:
				return [a, a1*aneg, b1*bneg]
			
def isprime(n):
	"""isprime(n) - Test whether n is prime using a variety of pseudoprime tests."""
	if (n in [2,3,5,7,11,13,17,19,23,29]): return True
	return isprimeE(n,2) and isprimeE(n,3) and isprimeE(n,5)
			
def isprimeF(n,b):
	"""isprimeF(n) - Test whether n is prime or a Fermat pseudoprime to base b."""
	return (pow(b,n-1,n) == 1)
	
def isprimeE(n,b):
	"""isprimeE(n) - Test whether n is prime or an Euler pseudoprime to base b."""
	if (not isprimeF(n,b)): return False
	r = n-1
	while (r % 2 == 0): r /= 2
	c = pow(b,r,n)
	if (c == 1): return True
	while (1):
		if (c == 1): return False
		if (c == n-1): return True
		c = pow(c,2,n)
	
def factor(n):
	"""factor(n) - Find a prime factor of n using a variety of methods."""
	if (isprime(n)): return n
	for fact in [2,3,5,7,11,13,17,19,23,29]:
		if n%fact == 0: return fact
	return factorPR(n)  # Needs work - no guarantee that a prime factor will be returned
			
def factors(n):
	"""factors(n) - Return a sorted list of the prime factors of n."""
	if (isprime(n)):
		return [n]
	fact = factor(n)
	if (fact == 1): return "Unable to factor "+str(n)
	facts = factors(n/fact) + factors(fact)
	facts.sort()
	return facts

def factorPR(n):
	"""factorPR(n) - Find a factor of n using the Pollard Rho method.
	Note: This method will occasionally fail."""
	for slow in [2,3,4,6]:
		numsteps=2*math.floor(math.sqrt(math.sqrt(n))); fast=slow; i=1
		while i<numsteps:
			slow = (slow*slow + 1) % n
			i = i + 1
			fast = (fast*fast + 1) % n
			fast = (fast*fast + 1) % n
			g = gcd(fast-slow,n)
			if (g != 1):
				if (g == n):
					break
				else:
					return g
	return 1
	
def eulerphi(n):
	"""eulerphi(n) - Computer Euler's Phi function of n - the number of integers
	strictly less than n which are coprime to n.  Otherwise defined as the order
	of the group of integers mod n."""
	thefactors = factors(n)
	thefactors.sort()
	phi = 1
	oldfact = 1
	for fact in thefactors:
		if fact==oldfact:
			phi = phi*fact
		else:
			phi = phi*(fact-1)
			oldfact = fact
	return phi
	
def carmichaellambda(n):
	"""carmichaellambda(n) - Computer Carmichael's Lambda function 
	of n - the smallest exponent e such that b**e = 1 for all b coprime to n.
	Otherwise defined as the exponent of the group of integers mod n."""
	thefactors = factors(n)
	thefactors.sort()
	thefactors += [0]  # Mark the end of the list of factors
	carlambda = 1 # The Carmichael Lambda function of n
	carlambda_comp = 1 # The Carmichael Lambda function of the component p**e
	oldfact = 1
	for fact in thefactors:
		if fact==oldfact:
			carlambda_comp = (carlambda_comp*fact)
		else:
			if ((oldfact == 2) and (carlambda_comp >= 4)): carlambda_comp /= 2 # Z_(2**e) is not cyclic for e>=3
			if carlambda == 1:
				carlambda = carlambda_comp
			else:
				carlambda = (carlambda * carlambda_comp)/gcd(carlambda,carlambda_comp)
			carlambda_comp = fact-1
			oldfact = fact
	return carlambda
	
def isprimitive(g,n):
	"""isprimitive(g,n) - Test whether g is primitive - generates the group of units mod n."""
	if gcd(g,n) != 1: return False  # Not in the group of units
	order = eulerphi(n)
	if carmichaellambda(n) != order: return False # Group of units isn't cyclic
	orderfacts = factors(order)
	oldfact = 1
	for fact in orderfacts:
		if fact!=oldfact:
			if pow(g,order/fact,n) == 1: return False
			oldfact = fact
	return True
	
def info():
	"""Return information about the module"""
	print locals()
	
# h / g^x_1 = (g^B)^x_0
# g^x_1 . q' = h (mod p) => xgcd(g^x_1, p) = [1, q, y] => h * xgcd(g^x_1, p) = [h, q', y'] 
def meet_in_the_middle(p, g, h, B):
	
	dict = {}
	#print B
	pow_g_x_1 = 1
	
	for x_1 in range(B + 1):
		[gcd, q1, q2] = xgcd(pow_g_x_1, p)	# (g^x_1) . q1 + (p) . q2 = gcd = 1 # (p is prime) => (g^x_1) . q1 = 1(mod p)
		d = (q1 * h) % p
		dict[d] = x_1			# (g^x_1) . (q1.h) + (p) . (q2.h) = h
		print(d)
		#dict[q1 * (h / gcd) % p] = x_1		# (g^x_1) . (q1.h) = h(mod p) => h / g^x_1 = q1.h (mod p)
		#print (pow_g_x_1 == powmod(g, x_1, p)) 
		pow_g_x_1 = (pow_g_x_1 * g) % p
	
	print('Done 1')
	
	#print len(dict)
	#for key, val in dict.items():
	#	print(str(key) + ': ' + str(val))
		
	pow_g_B = pow(g, B) % p
	pow_g_B_x_0 = 1
	
	for x_0 in range(B + 1):
		x_1 = dict.get(pow_g_B_x_0, None)
		print(pow_g_B_x_0)
		if x_1 != None:
			x = B * x_0 + x_1
			print (x_0, x_1)
			print (x)
			break
		#if (pow_g_B_x_0 != powmod(g, B * x_0, p)):
		#	print 'not equal'
		#	break
		pow_g_B_x_0 = (pow_g_B_x_0 * pow_g_B) % p
	
	print('Done 2')

def find_sol_naive(p, g, h):
	B = pow(2, 20)
	lb = 1
	ub = B
	#print B
	pow_g_x = 1
	while ub < B * B:
		for x_1 in range(lb, ub):
			if pow_g_x == h:
				print x
				break
			pow_g_x = (pow_g_x * g) % p
		print('Done')
		lb = ub
		ub = lb + B
#find_sol_naive(p, g, h)
		
p = 13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084171
g = 11717829880366207009516117596335367088558084999998952205599979459063929499736583746670572176471460312928594829675428279466566527115212748467589894601965568
h = 3239475104050450443565264378728065788649097520952449527834792452971981976143292558073856937958553180532878928001494706097394108577585732452307673444020333
B = pow(2, 20)

#print isprime(p)
#p = 11
#g = 3
#h = 5
#B = pow(2, 1)
meet_in_the_middle(p, g, h, B)