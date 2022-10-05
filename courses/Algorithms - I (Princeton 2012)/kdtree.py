#*************************************************************************
# * Name: Sandipan Dey
# * Email: sandipan.dey@gmail.com
# *
# *************************************************************************/

import math
import matplotlib.pylab as plt
from graphviz import Graph
from heapq import heappush, heappop, heapify
from random import random
from time import time
import datetime as dt
import os
from matplotlib.patches import Rectangle
import numpy as np

class Point2D:
    
    def __init__(self, x, y):
		self.x = x
		self.y = y

    def equals(self, p):
		return self.x == p.x and self.y == p.y
		
    def r(self): 
		return math.sqrt(self.x*self.x + self.y*self.y)

    def theta(self): 
		return math.atan2(self.y, self.x)

    def distanceTo(self, that):
        dx = self.x - that.x
        dy = self.y - that.y
        return math.sqrt(dx*dx + dy*dy)

    def distanceSquaredTo(self, that):
        dx = self.x - that.x
        dy = self.y - that.y
        return dx*dx + dy*dy

class RectHV:
    
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def width(self):
        return self.xmax - self.xmin

    def height(self):
        return self.ymax - self.ymin

    def intersects(self, that):
        return self.xmax >= that.xmin and self.ymax >= that.ymin and \
               that.xmax >= self.xmin and that.ymax >= self.ymin

    def distanceTo(self, p):
        return math.sqrt(self.distanceSquaredTo(p))

    def distanceSquaredTo(self, p):
		dx = 0.0 
		dy = 0.0
		if (p.x < self.xmin):
			dx = p.x - self.xmin
		elif (p.x > self.xmax):
			dx = p.x - self.xmax
		if (p.y < self.ymin):
			dy = p.y - self.ymin
		elif (p.y > self.ymax):
			dy = p.y - self.ymax
		return dx * dx + dy * dy
    
    def contains(self, p):
        return (p.x >= self.xmin) and (p.x <= self.xmax) and (p.y >= self.ymin) and (p.y <= self.ymax)

    def equals(self, that):
        if (self.xmin != that.xmin):
            return False
        if (self.ymin != that.ymin):
            return False
        if (self.xmax != that.xmax):
            return False
        if (self.ymax != that.ymax):
            return False
        return True

		
class PointSET:

	def __init__(self):
		self.pSet = set([])		

	def isEmpty(self): 
		return len(self.pSet) == 0

	def size(self): 
		return len(self.pSet)    

	def insert(self, p): 
		if not p in self.pSet:
			self.pSet.add(p)

	def contains(self, p):
		return p in pSet

	def range(self, rect):
		lst = []
		while p in self.pSet: 
			if rect.xmin <= p.x and rect.xmax >= p.x and rect.ymin <= p.y and rect.ymax >= p.y:
				lst.add(p)
		return lst

	def nearest(self, p): 
		mind, nearest = float('Inf'), None
		while q in self.pSet: 
			d = (p.x - q.x)**2  + (p.y - q.y)**2
			if (d < mind):
				mind, nearest = d, q
		return nearest   
 

class Vector:

    #def __init__(self, n):
    #    x = y = 0.
    
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    
    def plus(self, v):
    	return Vector(self.x + v.x, self.y + v.y)

    def minus(self, v):
    	return Vector(self.x - v.x, self.y - v.y)

    def distanceTo(self, v):
    	return math.sqrt((self.x - v.x)**2 +  (self.y - v.y)**2)

    def magnitude(self):
    	return math.sqrt(self.x*self.x + self.y*self.y)
    
    def direction(self):
    	return Vector(self.x/self.magnitude(), self.y/self.magnitude())
    
    def scale(self, F):
    	return Vector(self.x*F, self.y*F)
    
    def cartesian(self, index):
    	return self.x if index == 0 else self.y
		

class Node:

	def __init__(self, p, value, rect):
		self.p = p
		self.value = value
		self.rect = rect
		self.lb = self.rt = None			

class KdTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def size(self):
        return self.size

    def draw(self, xmin, xmax, ymin, ymax, n):
		plt.figure()
		self.draw2(self.root, 0, xmin, xmax, ymin, ymax)
		plt.savefig('test' + str(n) + '.png')  
		plt.close()
		#plt.show()

    def draw2(self, node, level, xmin, xmax, ymin, ymax): # draw all of the points to standard draw
		if node != None:
			plt.scatter(node.p.x, node.p.y, c='lightgreen', s=50)
			if level % 2 == 0:
				plt.plot((node.p.x, node.p.x), (ymin, ymax), 'r--')
			else:
				plt.plot((xmin, xmax), (node.p.y, node.p.y), 'b--')
			plt.text(node.p.x, node.p.y, str((round(node.p.x,3), round(node.p.y,3))), fontsize=5)
			self.draw2(node.lb, level + 1, xmin, xmax, ymin, ymax)
			self.draw2(node.rt, level + 1, xmin, xmax, ymin, ymax) 
			
    def to_html(self, level, str, nn=None):
		if level % 2 == 0:
			if nn:
				html = "'''<<table BORDER='2' color='GREEN'><tr><td>Vertical</td></tr><tr><td>" + str + "</td></tr></table>>'''"
			else:
				html = "'''<<table BORDER='2' color='BLUE'><tr><td>Vertical</td></tr><tr><td>" + str + "</td></tr></table>>'''"
		else:
			if nn:
				html = "'''<<table BORDER='2' color='GREEN'><tr><td>Horizontal</td></tr><tr><td>" + str + "</td></tr></table>>'''"
			else:
				html = "'''<<table BORDER='2' color='RED'><tr><td>Horizontal</td></tr><tr><td>" + str + "</td></tr></table>>'''"
		return html
	
    def draw1(self, n):
		g = Graph(name='input', node_attr={'shape': 'plaintext'}, format='png') #, node_attr={'shape': 'square', 'style': 'filled', 'color': 'lightblue2'})
		self.draw3(self.root, None, 0, g)
		g.render('test' + str(n), view=False)  

    def draw3(self, node, parent, level, g): # draw all of the points to standard draw
		if node != None:
			if level % 2 == 0:
				g.node('v' + str((node.p.x, node.p.y)), eval(self.to_html(level, str((round(node.p.x,3), round(node.p.y,3))))))
				if parent:
					g.edge('h' + str((parent.p.x, parent.p.y)), 'v' + str((node.p.x, node.p.y)))
			else:
				g.node('h' + str((node.p.x, node.p.y)), eval(self.to_html(level, str((round(node.p.x,3), round(node.p.y,3))))))
				if parent:
					g.edge('v' + str((parent.p.x, parent.p.y)), 'h' + str((node.p.x, node.p.y)))
			self.draw3(node.lb, node, level + 1, g)
			self.draw3(node.rt, node, level + 1, g) 
		
    def put(self, p, value):
        self.root = self.insert(self.root, p, value, 0)

    def insert(self, node, p, v, level):
	
		#self.draw1(self.size)
		#self.draw(-0.25, 1, -0.25, 1, self.size)
		if node == None:
			self.size += 1
			return Node(p, v, RectHV(p.x, p.y, p.x, p.y))
		elif node.p.equals(p):
			return node
			
		less = True
		if level % 2 == 0:
			less = p.x < node.p.x 
		else:
			less = p.y < node.p.y
			
		if (less):
			node.lb = self.insert(node.lb, p, v, level + 1)
			node.rect = RectHV(min(node.lb.rect.xmin, node.rect.xmin), min(node.lb.rect.ymin, node.rect.ymin), max(node.lb.rect.xmax, node.rect.xmax), max(node.lb.rect.ymax, node.rect.ymax))
		else:
			node.rt = self.insert(node.rt, p, v, level + 1)
			node.rect = RectHV(min(node.rt.rect.xmin, node.rect.xmin), min(node.rt.rect.ymin, node.rect.ymin), max(node.rt.rect.xmax, node.rect.xmax), max(node.rt.rect.ymax, node.rect.ymax))

		return node
    
    def contains(self, p):
		return self.get(self.root, p, 0) != None

    def get(self, p):
		n = self.get2(self.root, p, 0)
		return None if n == None else n.value
    
    def get2(self, node, p, level):
		if node == None:
			return None
		cmp = 0
		if level % 2 == 0:
			if p.x < node.p.x:
			   cmp = -1
			elif p.x > node.p.x:
				cmp = 1
		else:
			 if p.y < node.p.y:
				cmp = -1
			 elif p.y > node.p.y:
				 cmp = 1
		if cmp < 0:
			return self.get2(node.lb, p, level + 1)
		elif cmp > 0:
			return self.get2(node.rt, p, level + 1)
		else:
			if node.p.equals(p):
				return node
			else:
				return self.get2(node.rt, p, level + 1)

    def preOrder(self):
		preOrder2(self, self.root)

    def preOrder2(self, node):
		if node != None:
			print node.p.x + "," + node.p.y + ": " + node.rect
			self.preOrder2(node.lb)
			self.preOrder2(node.rt)

    def range(self, rect): # all points in the set that are inside the rectangle
		list = []
		if self.root != None:
			self.range2(self.root, list, rect)
		plt.figure()
		self.draw2(self.root, 0, -0.05, 1, -0.05, 1)
		currentAxis = plt.gca()
		currentAxis.add_patch(Rectangle((rect.xmin, rect.ymin), rect.xmax-rect.xmin, rect.ymax-rect.ymin, fill=None, alpha=1, edgecolor='black', linewidth='3'))
		for p in list:	
			plt.scatter(p.x, p.y, facecolor='yellow', edgecolor='black', linewidth='3', s=100)
		plt.savefig('test-range.png')  
		plt.close()
			
		return list

    def range2(self, node, list, query): # all points in the set that
        if node != None and node.rect.intersects(query):
            if (query.contains(node.p)):
				list += [node.p]
            self.range2(node.lb, list, query)
            self.range2(node.rt, list, query)

    def nearest(self, p, draw=False): # a nearest neighbor in the set to p; null if set is empty
		if self.root != None:
			return self.nearest2(self.root, p, float('Inf'), None, draw)
		return None

    def nearest2(self, node, query, minDist, nearest, draw=False, level=0):
	
		if draw:
			plt.figure()
			self.draw2(self.root, 0, -0.05, 1, -0.05, 1)
			plt.scatter(query.x, query.y, facecolor='white', edgecolor='black', linewidth='3', s=100)
			plt.scatter(node.p.x, node.p.y, facecolor='yellow', edgecolor='black', linewidth='3', s=100)
			plt.savefig('test' + str((node.p.x, node.p.y)) + '.png')  
			plt.close()
			#g = Graph(name='input', node_attr={'shape': 'plaintext'}, format='png') #, node_attr={'shape': 'square', 'style': 'filled', 'color': 'lightblue2'})
			#self.draw3(self.root, None, 0, g)
			#g.attr('node', style='filled', color='green')
			#g.node(('v' if level % 2 == 0 else 'h') + str((node.p.x, node.p.y)), eval(self.to_html(1, str((round(node.p.x,3), round(node.p.y,3))), True)))
			#g.render('test' + str((node.p.x, node.p.y)), view=False)  

		d = node.p.distanceSquaredTo(query)
		#print node.p.x, node.p.y, d, minDist, (None if nearest == None else (nearest.x, nearest.y))
		if (d < minDist):
			minDist, nearest = d, node.p
		if (node.lb != None and node.rt != None):
			d1 = node.lb.rect.distanceSquaredTo(query)
			d2 = node.rt.rect.distanceSquaredTo(query)
			if (d1 <= d2):
				if (d1 <= minDist):
					return self.nearest2(node.lb, query, minDist, nearest, draw, level+1)
				if (d2 <= minDist):
					return self.nearest2(node.rt, query, minDist, nearest, draw, level+1)
			else:
				if (d2 <= minDist):
					return self.nearest2(node.rt, query, minDist, nearest, draw, level+1)
				if (d1 <= minDist):
					return self.nearest2(node.lb, query, minDist, nearest, draw, level+1)
		elif (node.lb != None and node.lb.rect.distanceSquaredTo(query) <= minDist):
			return self.nearest2(node.lb, query, minDist, nearest, draw, level+1)
		elif (node.rt != None and node.rt.rect.distanceSquaredTo(query) <= minDist):
			return self.nearest2(node.rt, query, minDist, nearest, draw, level+1)
		return nearest
		
    def kNearest(self, p, k, draw=False):
		# null if set is empty
		if (self.root != None):
			kNearestNbrs = []
			self.knearest(self.root, p, float('Inf'), kNearestNbrs, min(self.size, k))
			if draw:
				g = Graph(name='input', node_attr={'shape': 'plaintext'}, format='png') #, node_attr={'shape': 'square', 'style': 'filled', 'color': 'lightblue2'})
				self.draw3(self.root, None, 0, g)
				for (_, p, l) in kNearestNbrs:
					#	print p.x, p.y
					g.attr('node', style='filled', color='green')
					g.node(('v' if l % 2 == 0 else 'h') + str((p.x, p.y)), eval(self.to_html(1, str((round(p.x,3), round(p.y,3))), True)))
				g.render('test_knn', view=False)  
			return [p for (_, p, _) in kNearestNbrs]
		return None
		
    def addToNearestNbrList(self, p, d, l, kNearestNbrs, k):
		if (len(kNearestNbrs) < k) or (d < abs(kNearestNbrs[0][0])):
			heappush(kNearestNbrs, (-d, p, l))
			if (len(kNearestNbrs) > k):
				heappop(kNearestNbrs)
		return kNearestNbrs

    def knearest(self, node, query, minDist, kNearestNbrs, k, level=0):
		d = node.p.distanceSquaredTo(query)
		kNearestNbrs = self.addToNearestNbrList(node.p, d, level, kNearestNbrs, k)
		if (len(kNearestNbrs) >= k and d < minDist):
			minDist = abs(kNearestNbrs[0][0])
		#print minDist
		if (node.lb != None and node.rt != None):
			d1 = node.lb.rect.distanceSquaredTo(query)
			d2 = node.rt.rect.distanceSquaredTo(query)
			if (d1 <= d2):
				if (d1 <= minDist):
					self.knearest(node.lb, query, minDist, kNearestNbrs, k, level+1)
				if (d2 <= minDist):
					self.knearest(node.rt, query, minDist, kNearestNbrs, k, level+1)
			else:
				if (d2 <= minDist):
					self.knearest(node.rt, query, minDist, kNearestNbrs, k, level+1)
				if (d1 <= minDist):
					self.knearest(node.lb, query, minDist, kNearestNbrs, k, level+1)
		elif (node.lb != None and node.lb.rect.distanceSquaredTo(query) <= minDist):
			self.knearest(node.lb, query, minDist, kNearestNbrs, k, level+1)
		elif (node.rt != None and node.rt.rect.distanceSquaredTo(query) <= minDist):
			self.knearest(node.rt, query, minDist, kNearestNbrs, k, level+1)

class Boid:    

    # Weights of a Boid's desires. Modify these and see what happens.

    BOID_AVOIDANCE_WEIGHT = 0.01
    HAWK_AVOIDANCE_WEIGHT = 0.01
    VELOCITY_MATCH_WEIGHT = 1
    PLUNGE_DEEPER_WEIGHT = 1
    RETURN_TO_ORIGIN_WEIGHT = 0.05

    # Agiility of a Boid is given by this value. Increase and they can react
    # more quickly (and also have a higher max velocity, due to simplicity
    # of physics model).

    THRUST_FACTOR = 0.0001

    # x,y stored as a Point2D
    # In the context of the Boid simulator, this is a little bit of
    # an awkward way to structure the code, since we map from Point2D 
    # to Boid -- i.e. they key is stored both in the symbol table
    # and in the value mapped to by the symbol table.

    # Despite this awkardness, this seems to be the best solution.

    def __init__(self, x=0., y=0., xvel=0., yvel=0.):
        self.position = Point2D(x, y)
        self.velocity = Vector(xvel, yvel)

    # Each Boid tries to avoid collisions with its neighbors. This method
    # provides a thrust vector to achieve that goal.
    def avoidCollision(self, neighbors):
        requestedVector = Vector(0.,0.)
        myPosition = Vector(self.position.x, self.position.y)
        # Sum the difference in position between this boid and its nearest
        # neighbors. Scale each vector so that closer boids are given more
        # weight.
        for b in neighbors:		
            neighborPosition = Vector(b.position.x, b.position.y)
            distanceTo = myPosition.distanceTo(neighborPosition)
            # don't count self
            if (distanceTo == 0.0): break #continue   #break
            avoidanceVector = myPosition.minus(neighborPosition)
            scaledAvoidanceVector = avoidanceVector.scale(1.0 / distanceTo)            
            requestedVector = requestedVector.plus(scaledAvoidanceVector)
        return requestedVector

	# Return a thrust vector to avoid a collision with the Hawk.
    def avoidCollision2(self, hawk):
		requestedVector = Vector(0.,0.)
		myPosition = Vector(self.position.x, self.position.y)
		hawkPosition = Vector(hawk.position.x, hawk.position.y)
		distanceTo = myPosition.distanceTo(hawkPosition)
		avoidanceVector = myPosition.minus(hawkPosition)
		scaledAvoidanceVector = avoidanceVector.scale(1.0 / distanceTo)            
		requestedVector = requestedVector.plus(scaledAvoidanceVector)
		return requestedVector

	# Return a thrust vector to match velocities with neighbors.
    def matchVelocity(self, neighbors):
		requestedVector = Vector(0.,0.)
		for b in neighbors:
			neighborVelocity = b.velocity
			matchingVector = neighborVelocity.minus(self.velocity)            
			requestedVector = requestedVector.plus(matchingVector)
		return requestedVector   

    # Return a thrust vector towards the center of the nearest neighbors.
    def plungeDeeper(self, neighbors):
        requestedVector = Vector(0.,0.)
        centroid = Vector(0.,0.)
        neighborCnt = 0
        for b in neighbors:
            neighborPosition = Vector(b.position.x, b.position.y)
            centroid = centroid.plus(neighborPosition)            
            neighborCnt += 1
        centroid = centroid.scale(1.0 / neighborCnt)
        myPosition = Vector(self.position.x, self.position.y)
        requestedVector = centroid.minus(myPosition)
        return requestedVector        
    
    # Return a thrust vector towards the origin: 0.5, 0.5
    def returnToWorld(self):
        requestedVector = Vector(0.,0.)
        center = Vector(0.5, 0.5)
        myPosition = Vector(self.position.x, self.position.y)
        requestedVector = center.minus(myPosition)
        return requestedVector        
    
    # Combines all thrust vectors into a single vector.
    # Each is weighted by arbitrary hard coded weights.
    def desiredAcceleration(self, neighbors, hawk):
        avoidanceVector = self.avoidCollision(neighbors).scale(Boid.BOID_AVOIDANCE_WEIGHT)
        hawkAvoidanceVector = self.avoidCollision2(hawk).scale(Boid.HAWK_AVOIDANCE_WEIGHT)
        matchingVector = self.matchVelocity(neighbors).scale(Boid.VELOCITY_MATCH_WEIGHT)
        plungingVector = self.plungeDeeper(neighbors).scale(Boid.PLUNGE_DEEPER_WEIGHT)
        returnVector = self.returnToWorld().scale(Boid.RETURN_TO_ORIGIN_WEIGHT)
        desired = Vector(0.,0.)
        desired = desired.plus(avoidanceVector)
        desired = desired.plus(hawkAvoidanceVector)
        desired = desired.plus(matchingVector)
        desired = desired.plus(plungingVector)
        desired = desired.plus(returnVector)
        if (desired.magnitude() == 0.0):
            return desired
        return desired.direction().scale(Boid.THRUST_FACTOR)
    
    def toString(self):
        return "" + x + " " + y + " " + " " + velocity

    # Updates position and velocity using rules given above.
    def updatePositionAndVelocity(self, neighbors, hawk):
        x = self.position.x + self.velocity.cartesian(0)
        y = self.position.y + self.velocity.cartesian(1)
        self.position = Point2D(x, y)
        desire = self.desiredAcceleration(neighbors, hawk)
        self.velocity = self.velocity.plus(desire)
        return desire


class Hawk:
   
    # create a hawk at (x, y) with zero velocity
    def __init__(self, x, y):
        self.position = Point2D(x, y)
        self.velocity = Vector(0.,0.)    
    
    # compare by y-coordinate, breaking ties by x-coordinate
    def compareTo(self, that):
        if (self.position.y < that.position.y): return -1
        if (self.position.y > that.position.y): return +1
        if (self.position.x < that.position.x): return -1
        if (self.position.x > that.position.x): return +1
        return 0
    
    # compare by y-coordinate, breaking ties by x-coordinate
    def distanceSquaredTo(self, that):
        return self.position.distanceSquaredTo(that.position)
    
    def returnToWorld(self):
        requestedVector = Vector(0.,0.)
        center = Vector(0.5, 0.5)
        myPosition = Vector(self.position.x, self.position.y)
        requestedVector = center.minus(myPosition)
        return requestedVector     
    
    def eatBoid(self, boid):
        requestedVector = Vector(0.,0.)
        boidPosition = Vector(boid.position.x, boid.position.y)
        myPosition = Vector(self.position.x, self.position.y)
        requestedVector = boidPosition.minus(myPosition)
        return requestedVector

    def updatePositionAndVelocity(self, nearest):
        x = self.position.x + self.velocity.cartesian(0)
        y = self.position.y + self.velocity.cartesian(1)
        position = Point2D(x, y)
        desire = self.eatBoid(nearest).direction().scale(0.0003)
        self.velocity = self.velocity.plus(desire)
        return desire

class BoidSimulator:
    
    #camera movement constants
            
    def lookUpBoids(self, bkd, points):
        values = []
        for p in points:
            values.append(bkd.get(p))
        return values
    
    def main(self):  
        
			NUM_BOIDS = 1000 #1500 #1000
			FRIENDS = 10 #15 #10

			hawk = Hawk(0.5, 0.3)
			
			# Each boid tracks a number of nearest neighbors equal to FRIENDS
			boids = [Boid() for _ in range(NUM_BOIDS)]

			radius = 0.5
			currentX = 0.5
			currentY = 0.5
			w = 2 #2 #5

			# Generate random boids.
			for i in range(NUM_BOIDS):
				startX = random()
				startY = random()
				velX = (random() - 0.5)/1000
				velY = (random() - 0.5)/1000
				boids[i] = Boid(startX, startY, velX, velY)
				
			timer = 0
			plt.rcParams['axes.facecolor'] = 'skyblue'
			while timer < 1000: #(True):

				timer += 1
				print timer
				#plt.clf()
				#plt.figure()
				# scale pen radius relative to zoom 
				#StdDraw.setPenRadius(0.01*(0.5/radius));
				#StdDraw.setXscale(currentX - radius, currentX + radius);
				#StdDraw.setYscale(currentY - radius, currentY + radius);

				# draw all boids and calculate their meanX and meanY
				meanX = 0
				meanY = 0
				for i in range(NUM_BOIDS):
					meanX += boids[i].position.x/NUM_BOIDS
					meanY += boids[i].position.y/NUM_BOIDS
					if timer % w == 0:
						plt.scatter(boids[i].position.x, boids[i].position.y, c='white', marker=r"$->>-$", s=20)
						plt.xticks([], []); plt.yticks([], [])

				# draw the hawk
				if timer % w == 0:
					plt.scatter(hawk.position.x, hawk.position.y, c='red', marker=r"$->>-$", s=75)

				# follow center of mass in tracking mode
				currentX = meanX
				currentY = meanY			

				# The entire KdTree must be rebuilt every frame. Since the boids
				# are random, we expect a roughly balanced tree, despite the
				# lack of balancing in KdTreeST.

				bkd = KdTree()
				for i in range(NUM_BOIDS):
					bkd.put(boids[i].position, boids[i])

				for i in range(NUM_BOIDS):
					kNearestPoints = bkd.kNearest(boids[i].position, FRIENDS)
					kNearest = self.lookUpBoids(bkd, kNearestPoints)
					boids[i].updatePositionAndVelocity(kNearest, hawk)

				# The hawk will chase the nearest boid.
				closestBoid = bkd.get(bkd.nearest(hawk.position))
				hawk.updatePositionAndVelocity(closestBoid)
				if timer % w == 0:
					plt.savefig('boid' + str(timer).zfill(4) + '.png')  
					plt.close()			
				#plt.show()

			
def main():

	'''
	file = "circle100.txt" #"circle10.txt" #"circle4.txt"
	input = open(file, "r")
	lines =  input.read().splitlines() #input.readlines()
	tree = KdTree()
	N = len(lines)
	for i in range(N):
	   #print lines[i].split(' ')
	   x, y = map(float, lines[i].split(' '))
	   p = Point2D(x, y)
	   tree.put(p, p)
	x, y = 0.3, 0.9 #random(), random()
	#x, y = 0.04, 0.7 #random(), random()
	print 'query:', (x, y)
	#p = tree.nearest(Point2D(x, y), True)
	#print (p.x, p.y)
	#for p in tree.kNearest(Point2D(x, y), 5, True): # 5 10 15
	#	print (p.x, p.y)
	# tree.range(RectHV(0.2, 0, 0.8, 1))
	print tree.size
	print tree.isEmpty()
	#tree.draw(-0.25, 1, -0.25, 1, 0)
	#tree.draw1(0)	
	'''
	
	'''
	for file in os.listdir("../"):
		if file.endswith(".txt"):
			input = open(os.path.join("../", file), "r")
			lines =  input.read().splitlines() #input.readlines()
			print file
			start = dt.datetime.utcnow() 
			tree = KdTree()
			#pset = PointSET()
			N = len(lines)
			for i in range(N):
			   #print lines[i].split(' ')
			   x, y = map(float, lines[i].split(' '))
			   p = Point2D(x, y)
			   tree.put(p, p)
			   #pset.insert(Point2D(x, y))
			end = dt.datetime.utcnow() 
			print (end - start).microseconds / 1000.
			start = dt.datetime.utcnow() 
			p = tree.nearest(Point2D(random(), random()))
			end = dt.datetime.utcnow() 
			print (end - start).microseconds / 1000.
			#print(p.x, p.y)
			#for d, p in tree.kNearest(Point2D(0, 0),2):
			#	print(d, p.x, p.y)
			for k in range(2, 100):
				start = dt.datetime.utcnow() 
				tree.kNearest(Point2D(random(), random()),k)
				end = dt.datetime.utcnow() 
				print k, (end - start).microseconds / 1000.
						
			#for p in tree.kNearest(Point2D(0, 0),2):
				#print(p.x, p.y)
			
	'''
	
	#BoidSimulator().main()
	
	#file = "input10K.txt"
	#input = open(file, "r")
	#lines =  input.read().splitlines() #input.readlines()
	tree = KdTree()
	N = 200 #len(lines)
	plt.figure()
	xs, ys, labels = [], [], []
	{-1:'r', +1:'b'}, {-1:'o', +1:'+'}
	for i in range(N):
		#print lines[i].split(' ')
		#x, y = map(float, lines[i].split(' '))
		x, y = random(), random()
		p = Point2D(x, y)
		label = 1 if random() > 0.5 else -1
		tree.put(p, (p, label))
		xs.append(x)
		ys.append(y)
		labels.append(label)
	xs, ys, labels = np.array(xs), np.array(ys), np.array(labels)
	#plt.scatter(xs, ys, c=[cols[label] for label in labels], marker=[mars[label] for label in labels], s=50)
	plt.scatter(xs[labels==+1], ys[labels==+1], color='blue', marker='+', s=5)
	plt.scatter(xs[labels==-1], ys[labels==-1], color='red', marker='o', s=5)
	print tree.size
	
	red = np.array([1,0,0,0.1])
	blue = np.array([0,0,1,0.1])
	k = 10 #5 #3
	for x in np.linspace(0, 1, 100):
		for y in np.linspace(0, 1, 100):
			#np.meshgrid(x, y)
			lst = []
			p = Point2D(x, y)
			for q in tree.kNearest(p, k):
				pt = tree.get(q)
				lst.append(pt[1])
			majority = max(lst, key=lst.count)
			#print p[0].x, p[0].y, p[1]
			if majority == -1:
				plt.scatter(p.x, p.y, color=red, marker='o')
			else:
				plt.scatter(p.x, p.y, color=blue, marker='+')
	
	plt.savefig('test_classification.png')  
	plt.close()
	
main()