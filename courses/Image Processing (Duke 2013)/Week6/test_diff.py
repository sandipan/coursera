# PDE diffusion.
#
# Diffusion equation 1 favours high contrast edges over low contrast ones.
# Diffusion equation 2 favours wide regions over smaller ones.

# Reference: 
# P. Perona and J. Malik. 
# Scale-space and edge detection using ansotropic diffusion.
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 
# 12(7):629-639, July 1990.

import numpy as np
from numpy import exp
import matplotlib.pylab as plt
import cv2

def diffusion(imfile, niter=50, kappa=10, lambda1=1/4., option=0):

	im = cv2.imread(imfile,0)
	print im.shape
	(rows,cols) = im.shape
	diff = im.astype(np.float)
  
	for i in range(niter):
	
	   print('iteration ' + str(i+1))

	   # Construct diffl which is the same as diff but
	   # has an extra padding of zeros around it.
	   diffl = np.zeros((rows+2, cols+2))
	   diffl[1:1+rows, 1:1+cols] = diff

	   # North, South, East and West differences
	   deltaN = diffl[0:rows,1:1+cols] - diff
	   deltaS = diffl[2:2+rows,1:1+cols] - diff
	   deltaE = diffl[1:1+rows,2:2+cols] - diff
	   deltaW = diffl[1:1+rows,0:cols] - diff

	   # Conduction
	   cN = cS = cE = cW = 1 # linear diffusion (gaussian blur)
	   if option == 1:
		 cN = exp(-(deltaN/kappa)**2)
		 cS = exp(-(deltaS/kappa)**2)
		 cE = exp(-(deltaE/kappa)**2)
		 cW = exp(-(deltaW/kappa)**2)
	   elif option == 2:
		cN = 1./(1 + (deltaN/kappa)**2)
		cS = 1./(1 + (deltaS/kappa)**2)
		cE = 1./(1 + (deltaE/kappa)**2)
		cW = 1./(1 + (deltaW/kappa)**2)

	   diff += lambda1*(cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
	   #cv2.imwrite("test/im" + str(i) + ".png", diff)
	   plt.imshow(diff, cmap='gray')
	   plt.title('Diffusion Iteration ' + str(i+1))
	   plt.savefig("test/diffusion_" + str(i).zfill(2) + imfile)   # save the figure to file
	   plt.close()
	plt.imshow(np.abs(im-diff), cmap='gray')
	plt.title('Diffusion Iteration ' + str(i+1))
	plt.savefig("test/diffusion_diff_" + str(i).zfill(2) + imfile)   # save the figure to file
	plt.close()

from scipy import signal, sparse
from numpy import pi, sqrt

def gauss_blur(imfile):
	im = cv2.imread(imfile,0)
	for s in range(1, 6):
		probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-5,6)] #np.random.normal(0, s, 11)
		kernel = np.outer(probs, probs)		
		out = signal.convolve2d(im, kernel, boundary='symm', mode='same')
		plt.imshow(out, cmap='gray')
		plt.title('Gaussian blur with sigma ' + str(s))
		plt.savefig("test/gauss_" + str(s) + imfile)   # save the figure to file
		plt.close()
		
def canny_edge(imfile):
	img = cv2.imread(imfile,0)
	edges = cv2.Canny(img,25,150)
	plt.imshow(edges, cmap='gray')
	plt.title('Edges with Canny Edge Detector')
	plt.savefig("test/canny_" + imfile)   # save the figure to file
	plt.close()

imfile = 'lena1.png' #'me.png' #'pami.png' #'me.png' #'ship1.png' #/'monument.png' #'ship1.png' #'gauss_4lena.png' #'lena.png' #'ship.png'  #'coin.png' # ##'Cameraman256.png' #'monument.png' #'lena.png'	   
#gauss_blur(imfile)
#diffusion(imfile, option=1, kappa=10)
#canny_edge(imfile)

def image_denoise_gradient_descent(imfile, lambda1=2, eta=0.1, niter=100):
	f = cv2.imread(imfile,0)
	(w,h) = f.shape
	u = f.astype(np.float) #+ np.random.rand(w,h)
	kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])		
	ll = []
	for i in range(niter):
		del_u = signal.convolve2d(u, kernel, boundary='symm', mode='same')		
		del_u = (f -  u) + lambda1*del_u	   
		u1 = u + eta*del_u
		l = np.linalg.norm(abs(u1 - u), 'fro')
		ll.append(l)
		print(l)
		u = u1 
		plt.imshow(u, cmap='gray')
		plt.title('Gradient Descent iteration ' + str(i))
		plt.savefig("test/denoise_grad_" + str(i).zfill(2) + imfile)   # save the figure to file
		plt.close()
	plt.imshow(np.abs(u-f), cmap='gray')
	plt.title('Gradient Descent iteration ' + str(i))
	plt.savefig("test/denoise_grad_diff_" + str(i) + imfile)   # save the figure to file
	plt.close()
	plt.plot(range(len(ll)), ll, '.-')
	plt.title('Gradient Descent Cost function decrease with iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Cost function')
	plt.savefig("test/denoise_grad_val_" + str(i) + imfile)   # save the figure to file
	plt.close()

def d_dx(I):
	(rows,cols) = I.shape
	I0 = np.zeros((rows+2, cols+2))
	I0[1:1+rows, 1:1+cols] = I
	return I0[2:2+rows,1:1+cols] - I

def d_dy(I):
	(rows,cols) = I.shape
	I0 = np.zeros((rows+2, cols+2))
	I0[1:1+rows, 1:1+cols] = I
	return I0[1:1+rows,2:2+cols] - I
	
def create_tampered_image(imfile, mask):
	f = cv2.imread(imfile,0)
	(w,h) = f.shape
	m = cv2.imread(mask,0)
	f &= m
	cv2.imwrite("tampered_" + mask.split('.')[0] + '_' + imfile, f)	

def create_random_mask(imfile, n_p=0.8):
	f = cv2.imread(imfile,0)
	(w,h) = f.shape
	m = np.random.rand(w,h)
	m[m >= n_p] = 1
	m[m < n_p] = 0
	m *= 255
	cv2.imwrite("noise_mask_" + imfile, m)	
	
def image_inpaint_gradient_descent(imfile, mask, lambda1=1, eta=0.001, niter=1000):
	
	D = cv2.imread(mask,0)
	D = D.astype(np.float) / 255
	f = cv2.imread(imfile,0)
	f = f.astype(np.float) / 255
	(w,h) = f.shape
	#R = np.random.random((w,h)) 
	#f = D * f + (1 - D) * R
	u = f
	#print(D)
	#print(R)
	laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])		
	grad_kernel_x = np.array([[-1, 1]])		
	grad_kernel_y = np.array([[-1], [1]])		
	
	ll = []
	for i in range(niter):
		
		u_x, u_y = signal.convolve2d(u, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u, grad_kernel_y, boundary='symm', mode='same')		
		u_xx, u_xy, u_yy = signal.convolve2d(u_x, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u_x, grad_kernel_y, boundary='symm', mode='same'), signal.convolve2d(u_y, grad_kernel_y, boundary='symm', mode='same')
		du = 2*lambda1*D*(f - u) + (u_xx*u_y**2 - 2*u_x*u_y*u_xy + u_yy*u_x**2) / (0.01+np.sqrt(u_x**2 + u_y**2)**3)		
		
		#del_u = signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same')		
		#u_x, u_y = signal.convolve2d(u, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u, grad_kernel_y, boundary='symm', mode='same')		
		#u_yx, u_xy = signal.convolve2d(u_y, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u_x, grad_kernel_y, boundary='symm', mode='same')	
		#du = (u_yx - u_xy) * del_u / np.sqrt(u_x**2 + u_y**2)
		#print np.linalg.norm(u_xy, 'fro')
		
		#del_u = signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same')		
		#del_u = cv2.filter2D(u, -1, laplace_kernel) 
		#du = D*(f -  u) + lambda1*del_u	   
				
		#u_x, u_y = signal.convolve2d(u, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u, grad_kernel_y, boundary='symm', mode='same')		
		#del_u = signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same') / (0.01 + np.sqrt(u_x**2 + u_y**2))	 	
		#del_u = -signal.convolve2d(signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same') / (0.01 + np.sqrt(u_x**2 + u_y**2)), laplace_kernel, boundary='symm', mode='same')
		#du = D*(f - u) + lambda1*del_u 
		#du = 2*lambda1*D*(f - u) + del_u / np.linalg.norm(np.sqrt(u_x**2 + u_y**2), 'fro')	   
		
		#d_dx_u, d_dy_u = d_dx(u), d_dy(u)
		#du = D*(f - u) + (d_dx(del_u)*d_dy_u - d_dy(del_u)*d_dx_u) / np.linalg.norm(np.sqrt(d_dx_u**2 + d_dy_u**2), 'fro')

		u1 = u + eta*du
		l = np.linalg.norm(abs(u1 - u), 'fro')
		ll.append(l)
		print(l)
		u = u1 
		if i % 4 == 0:
			plt.imshow(u, cmap='gray')
			plt.title('Gradient Descent iteration ' + str(i))
			plt.savefig("test/image_inpaint_" + str(i).zfill(3) + imfile)   # save the figure to file
			plt.close()
	
	#plt.imshow(np.abs(u-f), cmap='gray')
	plt.title('Gradient Descent iteration ' + str(i))
	plt.savefig("test/image_inpaint_" + str(i) + imfile)   # save the figure to file
	plt.close()
	plt.plot(range(len(ll)), ll, '.-')
	plt.title('Gradient Descent Cost function decrease with iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Cost function')
	plt.savefig("test/image_inpaint_" + str(i) + imfile)   # save the figure to file
	plt.close()	

#imfile = 'lena.png' #'elephant.jpg'	
#create_random_mask(imfile)
#imfile, mask = 'lena.png', 'noise_mask_lena.png' #'elephant.jpg', 'noise_mask_elephant.jpg' #'lena.png', 'mask.png'	   
#create_tampered_image(imfile, mask) 
#imfile, mask = 'im6.jpg', 'mask6.jpg'  #'tampered_lena.png', 'noise_mask_lena.png'  #'tampered_elephant.jpg', 'noise_mask_elephant.jpg' 
#'tampered_lena3.png', 'mask3.png' #'tampered_lena2.png', 'mask2.png' #'tampered_lena.png', 'mask.png' #'tampered_lena1.png', 'mask1.png' #'ellipses.png', 'mask4.png' ##'im1.png', 'mask4.png' #
#image_inpaint_gradient_descent(imfile, mask)
#https://in.mathworks.com/company/newsletters/articles/applying-modern-pde-techniques-to-digital-image-restoration.html

# http://www.boopidy.com/aj/cs/inpainting/
#def gradient_domain_editing():
	
# http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15463-f11/www/proj2/www/julenka/	

#import scipy

niter = 0
w, h = 0, 0

def callback(xk):
	global niter, w, h
	niter += 1
	print niter
	I = np.reshape(xk, (w,h))
	cv2.imwrite("pe_" + str(niter).zfill(3) + target, I*255)	

def poisson_editing(src, target, mask):

	global w, h
	
	S = cv2.imread(src,0).astype(np.float) / 255
	T = cv2.imread(target,0).astype(np.float) / 255
	M =  cv2.imread(mask,0).astype(np.float) / 255
	(w, h) = S.shape
	#A = np.zeros((w*h, w*h), dtype='float32')
	#A = sparse.coo_matrix((w*h, w*h), dtype=np.int8) # empty matrix
	b = np.zeros(w*h)
	
	row, col, data = [], [], []
	for x in range(w):
		for y in range(h):
			if M[x,y] > 0: # == 1:
				c = 0
				s, t = 0, 0
				if y < h-1: 
					row.append(h*x+y)
					col.append(h*x+y+1)
					data.append(1)
					#A[h*x+y, h*x+y+1] = 1   
					#b[h*x+y] += S[x,y+1] 
					s += S[x,y+1] 
					t += T[x,y+1] 
					c += 1
				if y > 0:   
					row.append(h*x+y)
					col.append(h*x+y-1)
					data.append(1)
					#A[h*x+y, h*x+y-1] = 1   
					#b[h*x+y] += S[x,y-1] 
					s += S[x,y-1] 
					t += T[x,y-1] 
					c += 1
				if x < w-1: 
					row.append(h*x+y)
					col.append(h*(x+1)+y)
					data.append(1)
					#A[h*x+y, h*(x+1)+y] = 1
					#b[h*x+y] += S[x+1,y]
					s += S[x+1,y]
					t += T[x+1,y]
					c += 1
				if x > 0:   
					row.append(h*x+y)
					col.append(h*(x-1)+y)
					data.append(1)
					#A[h*x+y, h*(x-1)+y] = 1 
					#b[h*x+y] += S[x-1,y] 
					s += S[x-1,y] 
					t += T[x-1,y] 
					c += 1
				row.append(h*x+y)
				col.append(h*x+y)
				data.append(-c)
				#A[h*x+y, h*x+y] = -c
				#b[h*x+y] += -c*S[x,y]
				s += -c*S[x,y]
				t += -c*T[x,y]
				#print s, t
				b[h*x+y] = s 
				#b[h*x+y] = s if abs(s) > abs(t) else t
			else:
				row.append(h*x+y)
				col.append(h*x+y)
				data.append(1)
				#A[h*x+y, h*x+y] = 1 
				b[h*x+y] = T[x,y]
			
	#I = np.reshape(np.linalg.solve(A, b), (w,h))
	#A = sparse.csr_matrix(A)
	A = sparse.coo_matrix((data, (row, col)), shape=(w*h, w*h), dtype=np.int8) # empty matrix
	#I = np.reshape(sparse.linalg.cg(A, b, maxiter=200, callback=callback)[0], (w,h))
	#I = np.reshape(sparse.linalg.cgs(A, b, maxiter=200, callback=callback)[0], (w,h))
	I = np.reshape(sparse.linalg.spsolve(A, b), (w,h))
	cv2.imwrite("pe_" + target, I*255)	

col = None	
def callback2(xk): 
	global niter, w, h, col
	I = np.reshape(xk, (w,h))
	cv2.imwrite("pe_col_" + str(niter).zfill(3) + '_' + str(col) + target, I*255)	
	niter += 1
	print niter

import os 
	
def poisson_editing_color(src, target, mask):

	global w, h, col, niter
	
	S = cv2.imread(src).astype(np.float) / 255
	T = cv2.imread(target).astype(np.float) / 255
	M =  cv2.imread(mask, 0).astype(np.float) / 255
	(w, h, d) = S.shape
	#A = np.zeros((w*h, w*h), dtype='float32')
	#A = sparse.coo_matrix((w*h, w*h), dtype=np.int8) # empty matrix
	b = np.zeros((w*h, d))
	
	row, col, data = [], [], []
	for x in range(w):
		for y in range(h):
			if M[x,y] > 0: # == 1:
				c = 0
				s, t = np.zeros(d, dtype=np.float), np.zeros(d, dtype=np.float)
				if y < h-1: 
					row.append(h*x+y)
					col.append(h*x+y+1)
					data.append(1)
					#A[h*x+y, h*x+y+1] = 1   
					s += S[x,y+1,:] 
					t += T[x,y+1,:] 
					c += 1
				if y > 0:   
					row.append(h*x+y)
					col.append(h*x+y-1)
					data.append(1)
					#A[h*x+y, h*x+y-1] = 1   
					s += S[x,y-1,:] 
					t += T[x,y-1,:] 
					c += 1
				if x < w-1: 
					row.append(h*x+y)
					col.append(h*(x+1)+y)
					data.append(1)
					#A[h*x+y, h*(x+1)+y] = 1
					s += S[x+1,y,:]
					t += T[x+1,y,:]
					c += 1
				if x > 0:   
					row.append(h*x+y)
					col.append(h*(x-1)+y)
					data.append(1)
					#A[h*x+y, h*(x-1)+y] = 1 
					s += S[x-1,y,:] 
					t += T[x-1,y,:] 
					c += 1
				row.append(h*x+y)
				col.append(h*x+y)
				data.append(-c)
				#A[h*x+y, h*x+y] = -c
				s += -c*S[x,y,:]
				t += -c*T[x,y,:]
				#print s, t
				b[h*x+y,:] = s
				# mixed
				#for dim in range(d):				
				#	b[h*x+y, dim] = s[dim] if abs(s[dim]) > abs(t[dim]) else t[dim]
			else:
				row.append(h*x+y)
				col.append(h*x+y)
				data.append(1)
				#A[h*x+y, h*x+y] = 1 
				b[h*x+y,:] = T[x,y,:]
			
	#I = np.reshape(np.linalg.solve(A, b), (w,h))
	#A = sparse.csr_matrix(A)
	A = sparse.coo_matrix((data, (row, col)), shape=(w*h, w*h), dtype=np.int8) # empty matrix
	#I = np.reshape(sparse.linalg.cg(A, b, maxiter=200, callback=callback)[0], (w,h))
	#I = np.reshape(sparse.linalg.cgs(A, b, maxiter=200, callback=callback)[0], (w,h))
	I = np.zeros((w, h, d))
	maxiter = 60
	for dim in range(d):
		I[:,:,dim] = np.reshape(sparse.linalg.spsolve(A, b[:,dim]), (w,h))
		#col = dim
		#niter = 0
		#I[:,:,dim] = np.reshape(sparse.linalg.cgs(A, b[:,dim], maxiter=maxiter, callback=callback2)[0], (w,h))
	cv2.imwrite("pe_" + target, I*255)		
	'''
	for i in range(maxiter):
		I = np.zeros((w, h, d))
		for dim in range(d):
			I[:,:,dim] = cv2.imread("pe_col_" + str(i).zfill(3) + '_' + str(dim) + target, 0)	
			os.remove("pe_col_" + str(i).zfill(3) + '_' + str(dim) + target)
		cv2.imwrite("pe_col_" + str(i).zfill(3) + target, I)		
	'''
	
def poisson_editing_color_mix(src, target, mask):

	global w, h, col, niter
	
	S = cv2.imread(src).astype(np.float) / 255
	T = cv2.imread(target).astype(np.float) / 255
	M =  cv2.imread(mask, 0).astype(np.float) / 255
	(w, h, d) = S.shape
	#A = np.zeros((w*h, w*h), dtype='float32')
	#A = sparse.coo_matrix((w*h, w*h), dtype=np.int8) # empty matrix
	#print(np.sum(M > 0))
	b = np.zeros((2*w*h, d))
	
	row, col, data = [], [], []
	for x in range(w):
		for y in range(h):
			if M[x,y] > 0: # == 1:
				s, t = np.zeros((2,d), dtype=np.float), np.zeros((2,d), dtype=np.float)
				if x < w-1: 
					row.append(2*(h*x+y))
					col.append(h*(x+1)+y)
					data.append(1)
					row.append(2*(h*x+y))
					col.append(h*x+y)
					data.append(-1)
					s[0,:] = S[x+1,y,:] - S[x,y,:]
					t[0,:] = T[x+1,y,:] - T[x,y,:]
				else:   
					row.append(2*(h*x+y))
					col.append(h*x+y)
					data.append(1)
					row.append(2*(h*x+y))
					col.append(h*(x-1)+y)
					data.append(-1)
					s[0,:] = S[x,y,:] - S[x-1,y,:]
					t[0,:] = T[x,y,:] - T[x-1,y,:]
				if y < h-1: 
					row.append(2*(h*x+y)+1)
					col.append(h*x+y+1)
					data.append(1)
					row.append(2*(h*x+y)+1)
					col.append(h*x+y)
					data.append(-1)
					s[1,:] = S[x,y+1,:] - S[x,y,:]
					t[1,:] = T[x,y+1,:] - T[x,y,:]
				else:   
					row.append(2*(h*x+y)+1)
					col.append(h*x+y)
					data.append(1)
					row.append(2*(h*x+y)+1)
					col.append(h*x+y-1)
					data.append(-1)
					s[1,:] = S[x,y,:] - S[x,y-1,:]
					t[1,:] = T[x,y,:] - T[x,y-1,:]
				#print s, t
				# mixed gradient
				for dim in range(d):				
					b[2*(h*x+y), dim] = s[0,dim] if sum(s[:,dim]**2) > sum(t[:,dim]**2) else t[0,dim]
					b[2*(h*x+y)+1, dim] = s[1,dim] if sum(s[:,dim]**2) > sum(t[:,dim]**2) else t[1,dim]
			else:
				row.append(2*(h*x+y))
				col.append(h*x+y)
				data.append(1)
				b[2*(h*x+y),:] = T[x,y,:]
			
	#I = np.reshape(np.linalg.solve(A, b), (w,h))
	#A = sparse.csr_matrix(A)
	A = sparse.coo_matrix((data, (row, col)), shape=(2*w*h, w*h), dtype=np.int8) # empty matrix
	#I = np.reshape(sparse.linalg.cg(A, b, maxiter=200, callback=callback)[0], (w,h))
	#I = np.reshape(sparse.linalg.cgs(A, b, maxiter=200, callback=callback)[0], (w,h))
	I = np.zeros((w, h, d))
	for iter in range(10, 200, 10) + [500, None]:
		print(iter)
		for dim in range(d):
			I[:,:,dim] = np.reshape(sparse.linalg.lsqr(A, b[:,dim], iter_lim=iter)[0], (w,h))		
		cv2.imwrite("pe_" + str(iter).zfill(4) + target, I*255)		
	
def texture_flattening(imfile, mask):
	global w, h, col, niter
	
	S = cv2.imread(src).astype(np.float) / 255
	M =  cv2.imread(mask, 0).astype(np.float) / 255 #+ 0.4
	(w, h, d) = S.shape
	b = np.zeros((2*w*h, d))
	
	row, col, data = [], [], []
	for x in range(w):
		for y in range(h):
			s, t = np.zeros((2,d), dtype=np.float), np.zeros((2,d), dtype=np.float)
			if x < w-1: 
				row.append(2*(h*x+y))
				col.append(h*(x+1)+y)
				data.append(1)
				row.append(2*(h*x+y))
				col.append(h*x+y)
				data.append(-1)
				s[0,:] = M[x,y] * (S[x+1,y,:] - S[x,y,:])
			else:   
				row.append(2*(h*x+y))
				col.append(h*x+y)
				data.append(1)
				row.append(2*(h*x+y))
				col.append(h*(x-1)+y)
				data.append(-1)
				s[0,:] = M[x,y] * (S[x,y,:] - S[x-1,y,:])
			if y < h-1: 
				row.append(2*(h*x+y)+1)
				col.append(h*x+y+1)
				data.append(1)
				row.append(2*(h*x+y)+1)
				col.append(h*x+y)
				data.append(-1)
				s[1,:] =  M[x,y] * (S[x,y+1,:] - S[x,y,:])
			else:   
				row.append(2*(h*x+y)+1)
				col.append(h*x+y)
				data.append(1)
				row.append(2*(h*x+y)+1)
				col.append(h*x+y-1)
				data.append(-1)
				s[1,:] =  M[x,y] * (S[x,y,:] - S[x,y-1,:])
			#print s, t
			# mixed gradient
			for dim in range(d):				
				b[2*(h*x+y), dim] = s[0,dim]
				b[2*(h*x+y)+1, dim] = s[1,dim]
			
	A = sparse.coo_matrix((data, (row, col)), shape=(2*w*h, w*h), dtype=np.int8) # empty matrix
	I = np.zeros((w, h, d))
	for iter in range(10, 200, 10) + [500, None]:
		print(iter)
		for dim in range(d):
			I[:,:,dim] = np.reshape(sparse.linalg.lsqr(A, b[:,dim], iter_lim=iter)[0], (w,h))		
		cv2.imwrite("pe_" + str(iter).zfill(4) + target, (I + 0.1)*255)		
	
##http://onsmith.web.unc.edu/assignment-2/	
#src = 'rainbow.png' #'face.png' #'me1.png' #'face1.png' # # #'peng.jpg' # #'madurga.jpg' #'srch.png' #'liberty.jpg' # #  #'sfruit.png' #'bear.bmp' #'source.jpg' # ## ###'bird.jpg' #'walking3.jpg' #'madurga.jpg' 
#target = 'sky1.png' #'mona.jpg' #'me.png' #'trekking.jpg' # #'kol.jpg' #'dsth.png' #'vic.jpg' # # #'dfruit.png' # 'waterpool.bmp' #'target.jpg' # # # # # #'cloud.jpg' # 'water-small.jpg' #'kol.jpg' # #
#mask = 'rmask.png' #'maskf.png' #'mme1.png' #'mface1.png' # # #'mask9.jpg' # #'maskp.jpg' #'maskh.png' #'mask10.jpg' # # #'fmask.png' #'mask.bmp' #'mask.jpg' # # # # #'mask8.jpg' # #'mask7.jpg' #'maskp.jpg' # #
#poisson_editing_color_mix(src, target, mask)	
#poisson_editing_color(src, target, mask)	
#poisson_editing(src, target, mask)	
#texture_flattening(imfile, mask)

#img = cv2.imread('me2.png') #'face1.png'
#edges = cv2.Canny(img,10,220)
#cv2.imwrite("mme2.png", edges)		 #mface1.png

#from matplotlib.colors import LogNorm

def curvature_image(imfile):
	u = cv2.imread(imfile,0).astype(np.float) / 255
	(m,n) = u.shape
	u1 = np.zeros((m+2, n+2))
	u1[1:1+m, 1:1+n] = u
	u_x = u1[2:2+m,1:1+n] - u
	u_y = u1[1:1+m,2:2+n] - u
	u_xx = u1[2:2+m,1:1+n] - 2*u + u1[0:m,1:1+n]
	u_yy = u1[1:1+m,2:2+n] - 2*u + u1[1:1+m,0:n]
	u_xy = (u1[2:2+m,2:2+n] + u1[0:m,0:n] - u1[0:m,2:2+n] - u1[2:2+m,0:n]) / 4.
	k = (u_xx*u_y**2 - 2*u_x*u_y*u_xy + u_yy*u_x**2) / (0.01 + (u_x**2 + u_y**2)**1.5)		
	plt.subplot(121), plt.imshow(u, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('The image')
	plt.subplot(122), plt.imshow(k, cmap='jet'), plt.xticks([]), plt.yticks([]), plt.colorbar(), plt.title('Curvature of the image')
	#plt.subplot(122), plt.imshow(k, cmap='jet', norm=LogNorm(vmin=np.min(k), vmax=np.max(k))), plt.xticks([]), plt.yticks([]), plt.colorbar(), plt.title('Curvature of the image')
	plt.savefig("cur_" +  imfile.split('.')[0] + '.png')   # save the figure to file
	plt.close()
	
#imfile = 'source.jpg' #'test.bmp'	# 
#curvature_image(imfile)

def add_salt_and_pepper(gb, prob):
	'''Adds "Salt & Pepper" noise to an image.
	gb: should be one-channel image with pixels in [0, 1] range
	prob: probability (threshold) that controls level of noise'''
	rnd = np.random.rand(gb.shape[0], gb.shape[1])
	noisy = gb.copy()
	noisy[rnd < prob] = 0
	noisy[rnd > 1 - prob] = 1
	return noisy

def TV(u):
	#grad_kernel_x = np.array([[-1, 1]])		
	#grad_kernel_y = np.array([[-1], [1]])		
	#u_x, u_y = signal.convolve2d(u, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u, grad_kernel_y, boundary='symm', mode='same')		
	(m,n) = u.shape
	u1 = np.zeros((m+2, n+2))
	u1[1:1+m, 1:1+n] = u
	u_x = u1[2:2+m,1:1+n] - u
	u_y = u1[1:1+m,2:2+n] - u
	gn = np.sqrt(u_x**2 + u_y**2)
	return gn, np.sum(gn)

def test_TV(imfile):	
	u = cv2.imread(imfile,0).astype(np.float) / 255
	ps, ts = np.linspace(0, 0.45, 46), []
	for p in ps:	
		u1 = add_salt_and_pepper(u, p)
		g, t = TV(u1)
		ts.append(t)
		print(p, t)
		plt.subplot(121), plt.imshow(u1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('image with salt-pepper, p = ' + str(np.round(p,2)))
		plt.subplot(122), plt.imshow(g, cmap='jet'), plt.xticks([]), plt.yticks([]), plt.title('gradient-norm (TV = ' + str(np.round(t, 2)) + ')') #plt.colorbar(), 
		plt.savefig("ng_" +  imfile.split('.')[0] + str(np.round(p,2)) + '.png')   # save the figure to file
		plt.close()
	plt.scatter(ps, ts, s=50), plt.xlabel('salt-pepper noise threshold prob'), plt.ylabel('total variance (TV)')
	plt.tight_layout()
	plt.savefig("tv_" +  imfile.split('.')[0] + '.png')   # save the figure to file
	plt.close()

def iso_diff(imfile): # heat
	#Parameters
	dt = 0.1;  #Time step
	T = 20;  #Stopping time
	K = 1;  #Conductivity
	u = cv2.imread(imfile,0).astype(np.float) / 255
	(m,n) = u.shape
	t = 0
	while t <= T:
		u1 = np.zeros((m+2, n+2))
		u1[1:1+m, 1:1+n] = u
		u_xx = u1[2:2+m,1:1+n] - 2*u + u1[0:m,1:1+n]
		u_yy = u1[1:1+m,2:2+n] - 2*u + u1[1:1+m,0:n]
		u += K * dt * (u_xx + u_yy)
		#u[0,:] = u[:,0] = u[m-1,:] = u[:,n-1] = 0 # Neumann BC 
		plt.imshow(u, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('isotropic diffusion (t = ' + str(np.round(t,1)) + ')')
		plt.savefig("iso_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '.png')   # save the figure to file
		plt.close()
		t += dt

def aniso_diff(imfile): 
	#Parameters
	dt = 0.1;  #Time step
	T = 20;  #Stopping time
	K = 1;  #Conductivity
	a = 100 #20 #5 #100 #5;
	u = cv2.imread(imfile,0).astype(np.float) #/ 255
	(m,n) = u.shape
	t = 0
	while t <= T:
		u1 = np.zeros((m+2, n+2))
		u1[1:1+m, 1:1+n] = u
		u_x = u1[2:2+m,1:1+n] - u
		u_y = u1[1:1+m,2:2+n] - u
		K = 1 / (1. + (u_x**2 + u_y**2)/a**2) # np.exp(-(u_x**2+u_y**2)/a**2)
		P, Q = K*u_x, K*u_y
		P1 = np.zeros((m+2, n+2))
		P1[1:1+m, 1:1+n] = P
		Q1 = np.zeros((m+2, n+2))
		Q1[1:1+m, 1:1+n] = Q
		u_t = (P - P1[0:m,1:1+n]) + (Q - Q1[1:1+m,0:n]) 
		u += K * dt * u_t
		#u[0,:] = u[:,0] = u[m-1,:] = u[:,n-1] = 0 # Neumann BC 
		plt.imshow(u, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('anisotropic diffusion (t = ' + str(np.round(t,1)) + ')')
		plt.savefig("aniso_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '_' + str(a) + '.png')   # save the figure to file
		plt.close()
		t += dt

# https://cvfxbook.com/video-lectures/
import matplotlib.image as mpimg		
def aniso_diff3(imfile): 
	#Parameters
	dt = 0.1;  #Time step
	T = 20;  #Stopping time
	K = 1;  #Conductivity
	a = 5 #10 #5 #100
	u = mpimg.imread(imfile).astype(np.float) # * 255 # / 255
	#print(u)
	#u[:,:,0], u[:,:,2] = u[:,:,2], u[:,:,0]
	(m,n,d) = u.shape
	print m,n,d
	t = 0
	#plt.imshow(u), plt.xticks([]), plt.yticks([]), plt.title('anisotropic diffusion (t = ' + str(np.round(t,1)) + ')')
	#plt.savefig("aniso_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '_' + str(a) + '.png')   # save the figure to file
	while t <= T:
		u1 = np.zeros((m+2, n+2, d))
		u1[1:1+m, 1:1+n, :] = u
		u_x = u1[2:2+m,1:1+n, :] - u
		u_y = u1[1:1+m,2:2+n, :] - u
		K = 1 / (1. + (u_x**2 + u_y**2)/a**2) #np.exp(-(u_x**2+u_y**2)/a**2) #
		P, Q = K*u_x, K*u_y
		P1 = np.zeros((m+2, n+2, d))
		P1[1:1+m, 1:1+n, :] = P
		Q1 = np.zeros((m+2, n+2, d))
		Q1[1:1+m, 1:1+n, :] = Q
		u_t = (P - P1[0:m,1:1+n, :]) + (Q - Q1[1:1+m,0:n, :]) 
		u += dt * u_t
		#u[0,:] = u[:,0] = u[m-1,:] = u[:,n-1] = 0 # Neumann BC 
		#plt.imshow(u.astype(np.uint8)), plt.xticks([]), plt.yticks([]), plt.title('anisotropic diffusion (t = ' + str(np.round(t,1)) + ')')
		#plt.savefig("aniso_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '_' + str(a) + '.png')   # save the figure to file
		#plt.close()
		t += dt
	plt.imshow(u.astype(np.uint8)), plt.xticks([]), plt.yticks([]), plt.title('anisotropic diffusion (t = ' + str(np.round(t,1)) + ')')
	plt.savefig("aniso_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '_' + str(a) + '.png')   # save the figure to file
	plt.close()
		
def add_gaussian_noise(u, mean=0, std=0.1):
	return u + np.random.normal(mean, std, u.shape)
	
def TV_denoise_linear(imfile, lambda1):
	#Parameters
	dt = 0.1;  #Time step
	T = 20;  #Stopping time
	u = cv2.imread(imfile,0).astype(np.float) / 255
	(m,n) = u.shape
	t = 0
	f = u
	while t <= T:
		u1 = np.zeros((m+2, n+2))
		u1[1:1+m, 1:1+n] = u
		u_x = u1[2:2+m,1:1+n] - u
		u_y = u1[1:1+m,2:2+n] - u
		u_xx = u1[2:2+m,1:1+n] - 2*u + u1[0:m,1:1+n]
		u_yy = u1[1:1+m,2:2+n] - 2*u + u1[1:1+m,0:n]
		u += dt*((u_xx + u_yy) - lambda1*(u-f))
		plt.imshow(u, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('TV denoise (t = ' + str(np.round(t,1)) + ')')
		plt.savefig("tv_denoise_linear_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '.png')   # save the figure to file
		plt.close()
		t += dt

def TV_denoise_nonlinear(imfile, lambda1):
	#Parameters
	dt = 1 #0.1;  #Time step
	T = 200;  #Stopping time
	u = cv2.imread(imfile,0).astype(np.float) #/ 255
	#print u
	(m,n) = u.shape
	t = 0
	f = u
	while t <= T:
		u1 = np.zeros((m+2, n+2))
		u1[1:1+m, 1:1+n] = u
		u_x = u1[2:2+m,1:1+n] - u
		u_y = u1[1:1+m,2:2+n] - u
		u_xx = u1[2:2+m,1:1+n] - 2*u + u1[0:m,1:1+n]
		u_yy = u1[1:1+m,2:2+n] - 2*u + u1[1:1+m,0:n]
		u_xy = (u1[2:2+m,2:2+n] + u1[0:m,0:n] - u1[0:m,2:2+n] - u1[2:2+m,0:n]) / 4.
		u += dt*(u_xx*u_y**2 - 2*u_x*u_y*u_xy + u_yy*u_x**2) / (0.1 + (u_x**2 + u_y**2)**1.5) - 2*lambda1*(u-f)	
		plt.imshow(u, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('TV denoise (t = ' + str(np.round(t,1)) + ')')
		plt.savefig("tv_denoise_nonlinear_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '.png')   # save the figure to file
		plt.close()
		t += dt
		
from copy import deepcopy		
def TV_inpainting(imfile, mask, lambda1):
	
	#Parameters
	dt = 0.25 #0.5 #0.25 #0.5;  #Time step
	T = 600 #100;  #Stopping time
	D = cv2.imread(mask,0).astype(np.float) / 255
	#print D
	#print(np.sum(D==0))
	f = cv2.imread(imfile,0).astype(np.float) / 255
	#print f
	(m,n) = f.shape
	R = np.random.random((m,n)) #* 255
	f = D * f + (1 - D) * R
	#print(R)
	
	t = 0
	u = deepcopy(f)
	uo = u
	
	plt.imshow(u, cmap='gray')
	plt.title('TV inpainting: Gradient Descent t = ' + str(t))
	plt.savefig("tv_inpaint_" +  '_' + ("%05.1f" % t) + imfile)   # save the figure to file
	plt.close()
		
	ll = []
	while t <= T:
	
		u1 = np.zeros((m+2, n+2))
		u1[1:1+m, 1:1+n] = u
		u_x = u1[2:2+m,1:1+n] - u
		u_y = u1[1:1+m,2:2+n] - u
		u_xx = u1[2:2+m,1:1+n] - 2*u + u1[0:m,1:1+n]
		u_yy = u1[1:1+m,2:2+n] - 2*u + u1[1:1+m,0:n]
		u_xy = (u1[2:2+m,2:2+n] + u1[0:m,0:n] - u1[0:m,2:2+n] - u1[2:2+m,0:n]) / 4.
		#u += (u_xx + u_yy)	- 2*lambda1*D*(u-f)	
		u += dt*((u_xx*u_y**2 - 2*u_x*u_y*u_xy + u_yy*u_x**2) / (0.01 + (u_x**2 + u_y**2)**1.5)	- 2*lambda1*D*(u-f))
		t += dt
		
		#del_u = signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same')		
		#u_x, u_y = signal.convolve2d(u, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u, grad_kernel_y, boundary='symm', mode='same')		
		#u_yx, u_xy = signal.convolve2d(u_y, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u_x, grad_kernel_y, boundary='symm', mode='same')	
		#du = (u_yx - u_xy) * del_u / np.sqrt(u_x**2 + u_y**2)
		#print np.linalg.norm(u_xy, 'fro')
		
		#del_u = signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same')		
		#del_u = cv2.filter2D(u, -1, laplace_kernel) 
		#du = D*(f -  u) + lambda1*del_u	   
				
		#u_x, u_y = signal.convolve2d(u, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(u, grad_kernel_y, boundary='symm', mode='same')		
		#del_u = signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same') / (0.01 + np.sqrt(u_x**2 + u_y**2))	 	
		#del_u = -signal.convolve2d(signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same') / (0.01 + np.sqrt(u_x**2 + u_y**2)), laplace_kernel, boundary='symm', mode='same')
		#du = D*(f - u) + lambda1*del_u 
		#du = 2*lambda1*D*(f - u) + del_u / np.linalg.norm(np.sqrt(u_x**2 + u_y**2), 'fro')	   
		
		#d_dx_u, d_dy_u = d_dx(u), d_dy(u)
		#du = D*(f - u) + (d_dx(del_u)*d_dy_u - d_dy(del_u)*d_dx_u) / np.linalg.norm(np.sqrt(d_dx_u**2 + d_dy_u**2), 'fro')

		l = np.linalg.norm(abs(uo - u), 'fro')
		ll.append(l)
		print(t, l)
		
		if t - int(t) == 0 and int(t) % 2 == 0:
			plt.imshow(u, cmap='gray')
			plt.title('TV inpainting: Gradient Descent t = ' + str(t))
			plt.savefig("tv_inpaint_" +  '_' + ("%05.1f" % t) + imfile)   # save the figure to file
			plt.close()
		
		uo = deepcopy(u) 

	plt.imshow(u, cmap='gray')
	plt.title('TV inpainting: Gradient Descent t =  ' + str(t))
	plt.savefig("tv_inpaint_" +  '_' + ("%05.1f" % t) + imfile)   # save the figure to file
	plt.close()
	plt.plot(range(len(ll)), ll, '.-')
	plt.title('Gradient Descent Cost function decrease with iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Cost function')
	plt.savefig("tv_inpaint_gd_" + imfile)   # save the figure to file
	plt.close()		
		
imfile = 'barbara.jpg' #'me2.png' #'Cameraman256.png' # # # # #'lena.png'
#test_TV(imfile)	
#iso_diff(imfile)
#aniso_diff(imfile)
#aniso_diff3(imfile)
#u = cv2.imread(imfile,0).astype(np.float) / 255
#u = add_gaussian_noise(u)
#print u.shape
#cv2.imwrite('noisy_cam.png', u*255)
#imfile = 'noisy_cam.png' #'lena.png'
#lambda1 = 1 #0.01 #1
#TV_denoise_linear(imfile, lambda1)
#TV_denoise_nonlinear(imfile, lambda1)

#imfile = 'Cameraman256.png'
#mask = 'text.png' #'cmask1.png'
#create_tampered_image(imfile, mask)
imfile, mask = 'tampered_lena2.png', 'mask2.png'
#imfile = 'ellipses.png' #'tampered_elephant.jpg' #'tampered_text_Cameraman256.png' #'tampered_cmask1_Cameraman256.png' # #
#mask = 'mask4.png' #'noise_mask_elephant.jpg' #'text.png' #'cmask1.png' #
#lambda1 = 0.05 #0.5 #0.01 #1
#TV_inpainting(imfile, mask, lambda1)
 
def image_denoise_gradient_descent_MAP_gaussian(imfile, lambda1=0.1, eta=0.1, s=3, niter=100):
	f = cv2.imread(imfile,0)
	(w,h) = f.shape
	ll = [None]*100 
	u = f.astype(np.float) #+ np.random.rand(w,h)
	kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])		
	for i in range(niter):
		del_u = signal.convolve2d(u, kernel, boundary='symm', mode='same')		
		del_u = (f -  u) / (2 * s * s) + lambda1*del_u	   
		u1 = u + eta*del_u
		l = np.linalg.norm(abs(u1 - u), 'fro')
		print(l)
		ll[i] = l
		u = u1 
		plt.subplot(121), plt.imshow(u, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Gradient Descent iteration ' + str(i))
		plt.subplot(122), plt.plot(range(len(ll)), ll), plt.plot(i, l, 'r.'), plt.xlim(0, 100), plt.ylim(0, 75), plt.xlabel('Iteration'), plt.ylabel('Cost function'), plt.tight_layout #, plt.title('Cost function for Gradient Descent: iteration ' + str(i))
		plt.savefig("test/map_denoise" + str(i).zfill(2) + imfile)   # save the figure to file
		plt.close()
	plt.imshow(np.abs(u-f), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Difference Image after Gradient Descent iteration ' + str(i))
	plt.savefig("test/map_denoise_diff" + str(i).zfill(2) + imfile)   # save the figure to file
	plt.close()
	
#image_denoise_gradient_descent_MAP_gaussian(imfile)	   

def image_deblur_gradient_descent(imfile, lambda1=0.1, eta=1, niter=100):
	f = cv2.imread(imfile,0)
	(w,h) = f.shape
	u = f.astype(np.float) #+ np.random.rand(w,h)
	s = 2
	probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-5,6)] #np.random.normal(0, s, 11)
	blur_kernel = np.outer(probs, probs)		
	laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])		
	grad_kernel_x = np.array([[-1, 1]])		
	grad_kernel_y = np.array([[-1], [1]])		
	for i in range(niter):
		del_u = -signal.convolve2d(signal.convolve2d(u, blur_kernel, boundary='symm', mode='same') - f, blur_kernel, boundary='symm', mode='same') 
		grad_u_x = signal.convolve2d(u, grad_kernel_x, boundary='symm', mode='same')		
		grad_u_y = signal.convolve2d(u, grad_kernel_y, boundary='symm', mode='same')		
		mod_grad_u = sqrt(grad_u_x*grad_u_x + grad_u_y*grad_u_y)
		del_u += signal.convolve2d(u, laplace_kernel, boundary='symm', mode='same') / np.linalg.norm(mod_grad_u, 'fro') # np.sum(mod_grad_u) #
		u1 = u + eta*del_u
		print np.linalg.norm(abs(u1 - u), 'fro')
		u = u1 
		plt.imshow(u, cmap='gray')
		plt.title('Gradient Descent iteration ' + str(i))
		plt.savefig("test/grad_" + str(i).zfill(2) + imfile)   # save the figure to file
		plt.close()

#image_deblur_gradient_descent(imfile)		

def image_sharpen(imfile, niter=1):
	f = cv2.imread(imfile,0)
	(w,h) = f.shape
	u = f.astype(np.float) #+ np.random.rand(w,h)
	n, s = 4, 1.4
	LOG_kernel = np.zeros((2*n+1,2*n+1))
	for x in range(-n, n+1):
		for y in range(-n, n+1):
			LOG_kernel[x,y] = int(-(10**3/(pi*s**4))*(1.-(x**2+y**2)/(2.*s**2))*exp(-(x**2+y**2)/(2.*s**2)))
	print LOG_kernel
	#laplace_kernel = np.array([[0, 1, 0], [1, -8, 1], [0, 1, 0]])	
	#conv = signal.convolve2d(f, laplace_kernel, boundary='symm', mode='same')
	#for i in range(niter):
	#	u += conv
	#	#u = 255 * (u - 0) / (np.max(u) - np.min(u))
	#	#u -= np.min(u)
	#	#u[u>255] = 255
	#	plt.imshow(u, cmap='gray')
	#	plt.title('Sharpening by adding Laplacian: iteration ' + str(i))
	#	plt.savefig("test/sharpen_" + str(i) + imfile)   # save the figure to file
	#	plt.close()
	kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	conv = cv2.filter2D(f, -1, kernel)
	#conv = signal.convolve2d(f, kernel, boundary='symm', mode='same')
	#print np.min(conv)
	#conv -= np.min(conv)
	#conv = conv = 255 * (conv - np.min(conv)) / (np.max(conv) - np.min(conv))
	#print np.min(conv), np.max(conv)
	#_, conv = cv2.threshold(conv,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	plt.imshow(conv, cmap='gray')
	plt.savefig("test/sharpen_" + imfile)   # save the figure to file
	plt.close
	
def unsharp_mask(imfile):
	image = cv2.imread(imfile, 0)
	gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
	unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
	plt.imshow(unsharp_image, cmap='gray')
	plt.title('Sharpening by unsharp mask')
	plt.savefig("test/unsharp_" + imfile)   # save the figure to file
	plt.close()

#image_sharpen(imfile)
#unsharp_mask(imfile)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def hog_circles(imfile):
	f = cv2.imread(imfile,0).astype(np.float)
	plt.imshow(f.T, cmap='gray')
	plt.show()
	laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])	
	f = signal.convolve2d(f, laplace_kernel, boundary='symm', mode='same')
	_, f = cv2.threshold(f,0,255,cv2.THRESH_BINARY)
	(w,h) = f.shape
	pixel_val, vote_thres = 255, 250 #275 
	r1, r2 = 10, 60
	print w,h
	votes = {}
	c, s = np.cos(np.linspace(0, 2*pi, 360)), np.sin(np.linspace(0, 2*pi, 360))
	for r in range(r1, r2):
		for x0 in range(r, w - r):
			print x0
			for y0 in range(r, h - r):
				xs, ys = x0 + (r*c).astype(np.int), y0 + (r*s).astype(np.int)
				total = sum([f[xs[i], ys[i]] == pixel_val for i in range(xs.shape[0])])
				if total >= vote_thres:
					votes[x0,y0,r] = total #votes.get((x0,y0,r),0) + 
	'''
	mpl.rcParams['legend.fontsize'] = 10
	x0, y0, r = range(w), range(h), range(r1, r2)
	fig = plt.figure()
	ax = Axes3D(fig)
	m, M = float('Inf'), float('-Inf')
	for (x0, y0, r) in votes:
		ax.scatter(x0, y0, r, c=votes[x0, y0, r], cmap='jet')
		#if m > votes[x0, y0, r]: m = votes[x0, y0, r]
		#if M < votes[x0, y0, r]: M = votes[x0, y0, r]
	#sm = plt.cm.ScalarMappable(cmap='jet')
	#sm = sm.set_clim(vmin=m, vmax=M)#, norm=plt.normalize(min=m, max=M))
	#plt.colorbar(sm)
	ax.legend()
	'''
	plt.imshow(f.T, cmap='gray')
	#plt.show()
	for (x0, y0, r) in votes:
		print x0, y0, r, votes[x0, y0, r]
		circle = plt.Circle((x0, y0), r)
		circle.set_facecolor('none')
		circle.set_edgecolor('r')
		plt.gcf().gca().add_artist(circle)
	plt.show()
	print len(votes), max(votes.values())
					
#hog_circles(imfile)
					
def image_op_gradient_descent(imfile, eta=0.1, niter=50):
	f = cv2.imread(imfile,0)
	(w,h) = f.shape
	u = f.astype(np.float)
	kernel_x = np.array([[-1, 1]])		
	kernel_y = np.array([[-1], [1]])		
	for i in range(niter):
		del_u_x = signal.convolve2d(u, kernel_x, boundary='symm', mode='same')		
		del_u_y = signal.convolve2d(u, kernel_y, boundary='symm', mode='same')		
		del_u = sqrt(del_u_x*del_u_x + del_u_y*del_u_y)
		u1 = u + eta*del_u
		#u1 = u - eta*del_u
		print np.linalg.norm(abs(u1 - u), 'fro')
		u = u1 
		plt.imshow(u, cmap='gray')
		plt.title('Gradient Descent iteration ' + str(i))
		plt.savefig("test/grad_" + str(i) + imfile)   # save the figure to file
		plt.close()

#image_op_gradient_descent(imfile)	   
		
'''
s, k = 1, 2
probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-2,3)] #np.random.normal(0, s, 11)
kernel = np.outer(probs, probs)
print kernel
#import matplotlib.pylab as plt
#plt.imshow(kernel)
#plt.colorbar()
#plt.show()
'''

from scipy.fftpack import dct, fft, idct, ifft

def jpeg(imfile):
	f = cv2.imread(imfile,0).astype(np.float)
	(w,h) = f.shape
	#print w,h
	ff = np.zeros((w,h))
	for i in range(w/8):
		for j in range(h/8):
			#print 8*i, 8*j
			ff[8*i:8*i+8, 8*j:8*j+8] = dct(f[8*i:8*i+8, 8*j:8*j+8])
	N = 2. #4. #8. #16. #256. #128. #64. #32.
	ff = N * np.round(ff / N)
	for i in range(w/8):
		for j in range(h/8):
			a = ff[8*i:8*i+8, 8*j:8*j+8]
			e = np.partition(a.flatten(), -8)[-8]
			a[np.where(a < e)] = 0
			ff[8*i:8*i+8, 8*j:8*j+8] = a
	for i in range(w/8):
		for j in range(h/8):
			#print 8*i, 8*j
			f[8*i:8*i+8, 8*j:8*j+8] = idct(ff[8*i:8*i+8, 8*j:8*j+8])
	
	#plt.imshow(ff.T, cmap='gray')
	plt.imshow(f.T, cmap='gray')
	plt.show()

imfile = 'lena.png'
#jpeg(imfile)	

# Seam Carving

def ComputeEnergy2(m, n, d, u):
	u1 = np.zeros((m+2, n+2, d))
	u1[1:1+m, 1:1+n, :] = u
	u_x = u1[2:2+m,1:1+n, :] - u1[1:1+m,1:1+n, :]
	u_y = u1[1:1+m,2:2+n, :] - u1[1:1+m,1:1+n, :]
	d_x = np.sum(abs(u_x), axis=2)
	d_y = np.sum(abs(u_y), axis=2)
	return d_x + d_y	
	
def ComputeEnergy(m, n, d, u):
	u1 = np.zeros((m+2, n+2, d))
	u1[1:1+m, 1:1+n, :] = u
	u1[0,1:1+n,:], u1[1+m,1:1+n,:] = u[2,0:n,:], u[0,0:n,:]
	u1[1:1+m,0,:], u1[1:1+m,1+n,:] = u[0:m,2,:], u[0:m,0,:]
	#print u1
	u_2x = u1[2:2+m,1:1+n, :] - u1[0:m,1:1+n, :]
	u_2y = u1[1:1+m,2:2+n, :] - u1[1:1+m,0:n, :]
	d_x2 = np.sum(u_2x**2, axis=2)
	d_y2 = np.sum(u_2y**2, axis=2)
	return np.sqrt(d_x2 + d_y2)	
	
def VerticalSeamCarve(m, n, E):
	C, B = [[0 for _ in range(n)] for _ in range(m)], [['' for _ in range(n)] for _ in range(m)] 
	for j in range(n):
		C[0][j] = E[0][j]
	for i in range(1, m):
		for j in range(0, n):
			minc, mind = C[i - 1][j], 'U'
			if j > 0 and C[i - 1][j - 1] < minc:
				minc, mind = C[i - 1][j - 1], 'L'
			if j < n - 1 and C[i - 1][j + 1] < minc:
				minc, mind = C[i - 1][j + 1], 'R'			
			C[i][j], B[i][j] = minc + E[i][j], mind
	
	(min_c, min_j) = min([(C[m-1][j], j) for j in range(n)])
	print min_c, min_j
	i, j = m - 1, min_j
	seam = []
	while i >= 0:
		#print E[i][j]
		seam.append((i,j))
		if B[i][j] == 'L':
			j -= 1
		elif B[i][j] == 'R':
			j += 1
		i -= 1	
	return seam	

def VerticalSeamCarve2(m, n, E, ns):
	C, B = [[0 for _ in range(n)] for _ in range(m)], [['' for _ in range(n)] for _ in range(m)] 
	for j in range(n):
		C[0][j] = E[0][j]
	for i in range(1, m):
		for j in range(0, n):
			minc, mind = C[i - 1][j], 'U'
			if j > 0 and C[i - 1][j - 1] < minc:
				minc, mind = C[i - 1][j - 1], 'L'
			if j < n - 1 and C[i - 1][j + 1] < minc:
				minc, mind = C[i - 1][j + 1], 'R'			
			C[i][j], B[i][j] = minc + E[i][j], mind
	
	seam = []
	for (min_c, min_j) in sorted([(C[m-1][j], j) for j in range(n)], reverse=True)[:ns]:
		print min_c, min_j
		i, j = m - 1, min_j
		while i >= 0:
			seam.append((i,j))
			if B[i][j] == 'L':
				j -= 1
			elif B[i][j] == 'R':
				j += 1
			i -= 1	
	return seam	


def SeamCarving2(imfile):
	oneM = 1024.*1024.
	I = mpimg.imread(imfile).astype(np.float) / 255 # * 255 # / 255
	print np.max(I)
	(m,n,d) = I.shape
	I = I[0:m,0:n,:]
	E = ComputeEnergy(m, n, d, I)
	plt.imshow(E, cmap='gray'); plt.xticks([]); plt.yticks([])
	plt.title('The Dual-Gradient Energy Function, image shape = (%dx%d)' % (m, n))
	plt.tight_layout()
	#plt.show()	
	plt.savefig("energy__" + imfile)   # save the figure to file
	plt.close()
	ns = 500
	vseam = VerticalSeamCarve2(m, n, E, ns)
	for (i,j) in vseam:
		I[i,j,:] = np.array([1,0,0])
	plt.imshow(I); plt.xticks([]); plt.yticks([])
	plt.tight_layout()
	#plt.show()	
	plt.savefig("seam__v" + imfile) # save the figure to file
	plt.close()			
	
	
def HorizontalSeamCarve(m, n, E):
	C, B = [[0 for _ in range(n)] for _ in range(m)], [['' for _ in range(n)] for _ in range(m)] 
	for i in range(m):
		C[i][0] = E[i][0]
	for j in range(1, n):
		for i in range(0, m):
			minc, mind = C[i][j - 1], 'U'
			if i > 0 and C[i - 1][j - 1] < minc:
				minc, mind = C[i - 1][j - 1], 'L'
			if i < m - 1 and C[i + 1][j - 1] < minc:
				minc, mind = C[i + 1][j - 1], 'R'			
			C[i][j], B[i][j] = minc + E[i][j], mind
	
	(min_c, min_i) = min([(C[i][n-1], i) for i in range(m)])
	print min_c, min_i
	i, j = min_i, n - 1
	seam = []
	while j >= 0:
		#print E[i][j]
		seam.append((i,j))
		if B[i][j] == 'L':
			i -= 1
		elif B[i][j] == 'R':
			i += 1
		j -= 1	
	return seam		

def RemoveVSeam(m, n, I, vseam):	
	for (i,j) in vseam:
		#print i, j, m, n
		I[i,j:n-1,:] = I[i,j+1:,:]
	return m, n-1, I	

def RemoveHSeam(m, n, I, hseam):	
	for (i,j) in hseam:
		I[i:m-1,j,:] = I[i+1:,j,:]
	return m-1, n, I	
	
def InsertVSeam(m, n, I, vseam):	
	I1 = np.zeros((m, n+1, 3))
	I1[:m, :n] = I 
	for (i,j) in vseam:
		#print i, j, m, n
		I1[i,j+1:,:] = I[i,j:,:]
		I1[i,j,:] = np.array([1,0,0])
	return m, n+1, I1		

def InsertVSeam2(m, n, I, vseam):	
	I1 = np.zeros((m, n+1, 3))
	I1[:m, :n] = I 
	for (i,j) in vseam:
		print i, j, m, n
		I1[i,j+1:,:] = I[i,j:,:]
		I1[i,j,:] =  (I[i,j-1,:] + I[i,j,:]) / 2. #if j > 0 else I[i,j,:] if j < n 
	return m, n+1, I1	
	
def insert_vertical_seams(I, k):
    I = I.copy()
    vseams = []
    for _ in range(k):
        (m,n,d) = I.shape
        E = compute_energy(m, n, d, I)
        vseam = vertical_seam_carve(m, n, E)
        vseams.append(vseam)
        I1 = np.zeros((m, n+1, 3))
        I1[:m, :n] = I 
        for (i,j) in vseam:
            #print (i, j, m, n)
            I1[i,j+1:,:] = I[i,j:,:]
            I1[i,j,:] =  (I[i,j-1,:] + I[i,j,:]) / 2 if j > 0 else I[i,j,:]
        I = I1
    return I1, vseams

def plot_vertical_seams_inserted(I, I_out, vseams, title):
    n = I.shape[1]
    I1 = np.zeros_like(I_out)    
    I1[:, :n] = I
    for vseam in vseams:
        for (i, j) in vseam:
            I1[i,j+1:,:] = I1[i,j:-1,:]
            I1[i,j,:] = np.array([1,0,0])
        #n += 1
    plt.figure(figsize=(20,7))
    plt.subplot(131), plt.imshow(I, aspect='auto'), plt.axis('off'), plt.title('input, size {}'.format(I.shape[:2]), size=20)
    plt.subplot(132), plt.imshow(I1, aspect='auto'), plt.axis('off'), plt.title('{} seam inserted'.format(len(vseams)), size=20)
    plt.subplot(133), plt.imshow(I_out, aspect='auto'), plt.axis('off'), plt.title('output, size {}'.format(I_out.shape[:2]), size=20)
    plt.suptitle(title, size=25)
    plt.tight_layout()
    plt.show()    

im_out, vseams = insert_vertical_seams(im, 2)
plot_vertical_seams_inserted(im, im_out, vseams, 'Increasing image size by inserting vertical seams')

def insert_vertical_seams(I, k):
    I = I.copy()
    m, n, d = I.shape
    I1 = np.zeros((m, n+k, d))    
    I, vseams = seam_carve_remove_seams(I, k)
    n = I.shape[1]
    I1[:, :n] = I
    for vseam in vseams[::-1]:
        for (i, j) in vseam:
            tmp = I1[i,j:-2,:]
            I1[i,j,:] = I1[i,j+1,:] = (I1[i,j-1,:] + I1[i,j+1,:]) / 2 #if j > 0 else I[i,j,:]
            I1[i,j+2:,:] = tmp
    #for l in range(k):
    #    m, n, d = I.shape
    #    I1 = np.zeros((m, n+2, 3))
    #    I1[:m, :n] = I 
    #    for (i,j) in vseams[::-1][l]:
    #        #print (i, j, m, n)
    #        I1[i,j+2:,:] = I[i,j:,:]
    #        I1[i,j,:] = I1[i,j+1,:] = (I[i,j-1,:] + I[i,j,:]) / 2 if j > 0 else I[i,j,:]
    #    I = I1
    return I1, vseams

def plot_vertical_seams_inserted(I, I_out, vseams, title):
    n = I.shape[1]
    I1 = np.zeros_like(I_out)    
    I1[:, :n] = I
    for vseam in vseams:
        for (i, j) in vseam:
            I1[i,j+1:,:] = I1[i,j:-1]
            I1[i,j,:] = np.array([1,0,0])
        #n += 1
    plt.figure(figsize=(20,7))
    plt.subplot(131), plt.imshow(I, aspect='auto'), plt.axis('off'), plt.title('input, size {}'.format(I.shape[:2]), size=20)
    plt.subplot(132), plt.imshow(I1, aspect='auto'), plt.axis('off'), plt.title('{} seam inserted'.format(len(vseams)), size=20)
    plt.subplot(133), plt.imshow(I_out, aspect='auto'), plt.axis('off'), plt.title('output, size {}'.format(I_out.shape[:2]), size=20)
    plt.suptitle(title, size=25)
    plt.tight_layout()
    plt.show()    

im = im_out
im_out2, vseams = insert_vertical_seams(im, 80)
plot_vertical_seams_inserted(im, im_out2, vseams, 'Increasing image size by inserting vertical seams')

id = 0

def plot_vertical_seams_removed(I0, I, I_out, vseams, title):
    global id
    vseams = vseams[::-1]
    n = I_out.shape[1]
    I1 = np.zeros_like(I)    
    I1[:, :n] = I_out
    for vseam in vseams:
        for (i, j) in vseam:
            I1[i,j+1:,:] = I1[i,j:-1,:]
            I1[i,j,:] = np.array([1,0,0])
        #n += 1
    plt.figure(figsize=(20,7))
    plt.subplot(141), plt.imshow(I0, aspect='auto'), plt.axis('off'), plt.title('input, size {}'.format(I.shape[:2]), size=20)
    plt.subplot(142), plt.imshow(I, aspect='auto'), plt.axis('off'), plt.title('input (with mask), size {}'.format(I.shape[:2]), size=20)
    plt.subplot(143), plt.imshow(I1, aspect='auto'), plt.axis('off'), plt.title('{} seams removed'.format(len(vseams)), size=20)
    plt.subplot(144), plt.imshow(I_out, aspect='auto'), plt.axis('off'), plt.title('output, size {}'.format(I_out.shape[:2]), size=20)
    plt.suptitle(title, size=25)
    plt.tight_layout()
    plt.savefig('1seam_{:03d}.png'.format(id))
    plt.close()
    id += 1

def seam_carving_remove_vertical_seam_with_mask(I, M):
    (m,n,d) = I.shape
    E = compute_energy(m, n, d, I)
    E *= M[:,:,0]
    vseam = vertical_seam_carve(m, n, E)
    _, _, I_out = remove_vertical_seam(m, n, I, vseam)
    m, n, M = remove_vertical_seam(m, n, M, vseam)
    return I_out, M, vseam

def seam_carve_remove_seams_with_mask(I, M, k):
    I1, M1 = I, M
    I = I.copy()
    M = M.copy()
    #M[M == 0] = -1 * 255
    M[(M[...,0]==1) & (M[...,1]==0) & (M[...,2]==0)] = -99999 #-1*255
    M[(M[...,0]==0) & (M[...,1]==1) & (M[...,2]==0)] = 10**9 #1*255
    #I *= M
    #I[I < 0] = -1
    #I[I > 1] = 2
    #print(I.max(), I.min())
    #m, n = I.shape[:2]
    vseams = []
    i = 0
    while i < k:
        I, M, vseam = seam_carving_remove_vertical_seam_with_mask(I, M)
        vseams.append(vseam)
        plot_vertical_seams_removed(I1, 0.4*I1 + 0.6*M1, I, vseams, 'Protecting (green mask) and Removing object (red mask) with seam carving')
        i += 1
    return I, vseams

im, mask = mpimg.imread('man_dog.jpeg')[...,:3], mpimg.imread('man_dog_mask.jpg')[...,:3]
im, mask = im / im.max(), mask / mask.max()
#mask[np.mean(mask, axis=2) > 0.05] = 1
mask2 = np.ones_like(mask)
mask2[(mask[...,0]>0.95) & (mask[...,1]<0.05) & (mask[...,2]<0.05)] = [1,0,0]
mask2[(mask[...,0]<0.05) & (mask[...,1]>0.95) & (mask[...,2]<0.05)] = [0,1,0]
    
print(im.shape, mask.shape)
im_out, vseams = seam_carve_remove_seams_with_mask(im, mask2, 120)
	
# http://www.cs.bu.edu/fac/betke/cs585/restricted/papers/Avidan-SeamCarving-2007.pdf	
# http://mmlab.ie.cuhk.edu.hk/archive/gbq/project4.htm
from sys import getsizeof
def SeamCarving(imfile, mask=None):
	#I = np.ones((4,3,3))*255
	#I[:,:,1] = np.array([[101]*3, [153]*3, [203,204,205], [255]*3])		
	#I[:,:,2] = np.array([[51,153,255] for _ in range(4)])
	#print I		
	#print ComputeEnergy(4, 3, 3, I)
	#E = [[240.18,225.59,302.27,159.43,181.81,192.99],[124.18,237.35,151.02,234.09,107.89,159.67],[111.10,138.69,228.10,133.07,211.51,143.75],[130.67,153.88,174.01,284.01,194.50,213.53],[179.82,175.49,70.06,270.80,201.53,191.20]]		
	#m, n = len(E), len(E[0])
	#vseam = VerticalSeamCarve(m, n, E)
	#I = cv2.imread(imfile).astype(np.float)
	oneK = 1024.
	oneM = oneK*oneK
	I = mpimg.imread(imfile).astype(np.float)[:,:,:3] / 255 #* 255 # / 255
	print np.max(I)
	(m,n,d) = I.shape
	M = None
	if mask:
		M = mpimg.imread(mask).astype(np.float)[:,:,:3] / 255 #
		print I.shape, M.shape
		M[M == 0] = -1 * 255
		I *= M
		I[I < 0] = -1
		plt.imshow(I); plt.xticks([]); plt.yticks([])
		plt.title('Masked Image')
		plt.tight_layout()
		plt.savefig("masked_" + imfile)   # save the figure to file
		plt.close()

	vseams = []
	vseamspixels = []
	
	for iter in range(100): # 300
		
		print iter, m, n, I.shape
		
		'''
		I = deepcopy(I[0:m,0:n,:])
		E = ComputeEnergy(m, n, d, I)
		hseam = HorizontalSeamCarve(m, n, E)
		I = deepcopy(I)
		for (i,j) in hseam:
			I[i,j,:] = np.array([1,0,0])
		plt.imshow(I); plt.xticks([]); plt.yticks([])
		plt.title('Removing H-Seam, image shape = (%dx%d), size = %.2f MB' % (m,n, getsizeof(I)/oneM))
		plt.tight_layout()
		#plt.show()
		plt.savefig("seam_" + str(iter).zfill(3) + '_h' + imfile)   # save the figure to file
		plt.close()
		m, n, I = RemoveHSeam(m, n, I, hseam)
		'''
		
		I = deepcopy(I[0:m,0:n,:])
		E = ComputeEnergy(m, n, d, I)
		if mask:
			print I.shape, E.shape, M[:,:,0].shape
			M = deepcopy(M[0:m,0:n,:])
			E *= M[:,:,0]
		
		'''
		plt.imshow(E, cmap='gray'); plt.xticks([]); plt.yticks([])
		plt.title('The Dual-Gradient Energy Function, image shape = (%dx%d)' % (m, n))
		plt.tight_layout()
		#plt.show()	
		plt.savefig("energy_" + str(iter).zfill(3) + imfile)   # save the figure to file
		plt.close()
		'''
		
		vseam = VerticalSeamCarve(m, n, E)
		vseams.append(vseam)
		#pixels = np.zeros(m, 3)
		#for i in range(len(vseam)): 
		#	pixels[i, :] = I[vseam[i][0], vseam[i][1], :]
		#vseamspixels.append(pixels)
		#I1 = deepcopy(I)
		for (i,j) in vseam:
			I[i,j,:] = np.array([1,0,0])
			
		plt.imshow(I); plt.xticks([]); plt.yticks([])
		plt.title('Removing V-Seam, image shape = (%dx%d), size = %0.2f MB' % (m, n, getsizeof(I)/oneM))
		plt.tight_layout()
		#plt.show()	
		plt.savefig("seam_" + str(iter).zfill(3) + '_v' + imfile)   # save the figure to file
		plt.close()
		
		m, n, I = RemoveVSeam(m, n, I, vseam)		
		if mask:
			_, _, M = RemoveVSeam(m, n+1, M, vseam)	
	
	'''	
	I = deepcopy(I[0:m,0:n,:])
	for i in range(99, -1, -1): # 300
		print i, m, n, I.shape
		m, n, I = InsertVSeam(m, n, I, vseams[i])
		#m, n, I = InsertVSeam2(m, n, I, vseams[i])
		plt.imshow(I); plt.xticks([]); plt.yticks([])
		plt.title('Inserting V-Seam, image shape = (%dx%d), size = %0.2f MB' % (m, n, getsizeof(I)/oneM))
		plt.tight_layout()
		#plt.show()	
		plt.savefig("iseam_" + str(100-i).zfill(3) + '_v' + imfile)   # save the figure to file
	'''
		
def InsertVSeam1(m, n, I, vseam):	
	I1 = np.zeros((m, n+1, 3))
	I1[:m, :n] = I 
	for (i,j) in vseam:
		#print i, j, m, n
		I1[i,j+1:,:] = I[i,j:,:]
		I1[i,j,:] = (I[i,j-1,:] + I[i,j,:]) / 2. if j > 0 else I[i,j,:]
	return m, n+1, I1			

def SeamCarving1(imfile):
	oneK = 1024.
	oneM = oneK*oneK
	I = mpimg.imread(imfile).astype(np.float)[:,:,:3] #* 255 # / 255
	print np.max(I)
	(m,n,d) = I.shape
	
	for iter in range(200): # 100
		
		print iter, m, n, I.shape
		
		E = ComputeEnergy(m, n, d, I)
		
		vseam = VerticalSeamCarve(m, n, E)
		I1 = deepcopy(I)
		for (i,j) in vseam:
			I1[i,j,:] = np.array([1,0,0])
		plt.imshow(I1); plt.xticks([]); plt.yticks([])
		plt.title('Removing V-Seam, image shape = (%dx%d), size = %0.2f MB' % (m, n, getsizeof(I1)/oneM))
		plt.tight_layout()
		#plt.show()	
		plt.savefig("seam_" + str(iter).zfill(3) + '_v' + imfile)   # save the figure to file
		plt.close()
		m, n, I = InsertVSeam1(m, n, I, vseam)		
					
#imfile = 'sea.jpg' #'dolphin.jpg' #'vr.jpg' #'house.png' # #'temple.jpg' #'cycle.png' #'bird2.png' #'dolphin.jpg' #'man.png' # 'shoes.jpg' #'Fuji.jpg' # # # # #'sea2.png' # # # #'HJoceanSmall.png'
#maskfile = 'seamask.jpg' #'manmask.png' #'shoemask.jpg'
#SeamCarving(imfile, maskfile)
#SeamCarving2(imfile)
#SeamCarving1(imfile)

from numpy import pi, exp, sqrt, trace
from numpy.linalg import eig, det
'''
The function to compute a (2*k+1)x(2*k+1) Gaussian kernel with mean 0 and standard deviation s
'''
def gaussian_kernel(k, s = 0.5):
	# generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
	probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
	return np.outer(probs, probs)

#from PIL import Image, ImageDraw
def make_rectangle(l, w, theta, offset=(0,0)):
    c, s = np.cos(theta), np.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return np.array([[c*x-s*y+offset[0], s*x+c*y+offset[1]] for (x,y) in rectCoords]).astype(np.int32)

# a = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])	

'''
The function to compute the simple (normalized) descriptor for a Harris Corner point (x,y) for an image I with the point in the center of the descriptor
'''
def get_descriptor(I, x, y):
	m, n = 8, 8	
	d = np.array([I[x-m/2:x+m/2+1,y-n/2:y+n/2+1,0].ravel(), I[x-m/2:x+m/2+1,y-n/2:y+n/2+1,1].ravel(), I[x-m/2:x+m/2+1,y-n/2:y+n/2+1,2].ravel()])
	return (d - np.mean(d, axis = 0)) / np.std(d, axis=0)
	#return (d - np.array([np.mean(d[:,:,0]), np.mean(d[:,:,1]), np.mean(d[:,:,2])])) / np.array([np.std(d[:,:,0]), np.std(d[:,:,1]), np.std(d[:,:,2])])

# http://www.cs.cornell.edu/courses/cs4670/2015sp/projects/pa2/
# http://www.connellybarnes.com/work/class/2017/intro_vision/proj1/	
'''
The function to compute the simple (normalized) descriptor for a Harris Corner point (x,y) for an image I with the point in the center of the descriptor
'''
def compute_harris_corner(imfile, threshold=10**(-4)):
	
	# 1. compute the gradient matrices and the matrix R to to store the corner-strengths
	
	# 2. use the threshold to get rid of the weak corner features
	
	# 3. use non-maximum supression to compute local maximum of the features and discard others
	
	# 4. compute the descriptors for each of the remaining feature points with get_descriptor()
	return None
	
def HarrisCorner(imfile, threshold=10**(-4)): #10**(-4)):
	I = cv2.imread(imfile,0).astype(np.float) / 255.
	(m, n) = I.shape
	print m, n, np.max(I)
	grad_kernel_x = np.array([[-1, 1]])		
	grad_kernel_y = np.array([[-1], [1]])
	Ix, Iy = signal.convolve2d(I, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(I, grad_kernel_y, boundary='symm', mode='same')		
	mI, dI = np.sqrt(Ix**2 + Iy**2), np.arctan(Iy/(Ix+0.01))
	R = np.zeros((m, n))
	L1 = np.zeros((m, n))
	#L2 = np.zeros((m, n))
	k = 4 #2
	kappa = 0.02
	gauss_kernel = 	gaussian_kernel(k)
	I = signal.convolve2d(I, gauss_kernel, boundary='symm', mode='same')
	for i in range(k, m-k):
		#print i
		for j in range(k, n-k):
			#C = np.array([[np.mean(gauss_kernel * Ix[i-k:i+k+1,j-k:j+k+1]**2), np.mean(gauss_kernel * Ix[i-k:i+k+1,j-k:j+k+1] * Iy[i-k:i+k+1,j-k:j+k+1])],  \
			#			  [np.mean(gauss_kernel * Ix[i-k:i+k+1,j-k:j+k+1] * Iy[i-k:i+k+1,j-k:j+k+1]), np.mean(gauss_kernel * Iy[i-k:i+k+1,j-k:j+k+1]**2)]])
			C = np.array([[np.mean(Ix[i-k:i+k+1,j-k:j+k+1]**2), np.mean(Ix[i-k:i+k+1,j-k:j+k+1] * Iy[i-k:i+k+1,j-k:j+k+1])],  \
						  [np.mean(Ix[i-k:i+k+1,j-k:j+k+1] * Iy[i-k:i+k+1,j-k:j+k+1]), np.mean(Iy[i-k:i+k+1,j-k:j+k+1]**2)]])
			R[i, j] = det(C) - kappa*trace(C)**2
			w, _ = eig(C)
			L1[i, j] = min(w) #w[0]
			#L2[i, j] = w[1]
	print np.max(R), np.min(R), threshold	
	R[R < threshold] = 0
	
	'''
	plt.subplot(121); plt.imshow(R, cmap='gray'); plt.xticks([]); plt.yticks([]) #plt.imshow(L2, cmap='gray')
	plt.subplot(122); plt.imshow(L1, cmap='gray'); plt.xticks([]); plt.yticks([])
	plt.tight_layout()
	#plt.show()
	plt.savefig(("hc1_%.05f" % (threshold)) + imfile)   # save the figure to file
	plt.close()
	'''
	
	#indices = np.dstack(np.unravel_index(np.argsort(R.ravel())[::-1], (m, n)))[0, :, :]
	indices = np.dstack(np.unravel_index(np.argsort(-R.ravel()), (m, n)))[0, :, :]
	Rs = []
	max_index = 0
	for i,j in indices:
		if R[i,j] == 0:
			break
		Rs.append(R[i,j])
		max_index += 1
	indices = indices[:max_index,:]	
	
	'''
	plt.plot(range(max_index), Rs[:max_index])
	#plt.show()
	plt.savefig("hist_" + imfile)   # save the figure to file
	'''
	#print indices.shape #, indices
	
	#lm_indices = indices
	#print lm_indices
	lm_indices = []
	found = set([])
	for i in range(indices.shape[0]):
		x, y = indices[i,0], indices[i,1]
		if not (x,y) in found: # and R[x,y] > 0:
			lm_indices.append([x,y])	
		for i in range(-k, k+1):
			for j in range(-k, k+1):
				found.add((x+i,y+j))
	lm_indices = np.array(lm_indices)
	#print lm_indices.shape #, lm_indices
	
	descriptor = []
	I = cv2.imread(imfile).astype(np.float) / 255
	for i in range(lm_indices.shape[0]):
		x, y = lm_indices[i,0], lm_indices[i,1]
		#rpts = make_rectangle(100*mI[x,y], 100*mI[x,y], dI[x,y], (y, x))
		#rpts = make_rectangle(10, 10, dI[x,y], (y, x))
		rpts = make_rectangle(25*mI[x,y], 25*mI[x,y], dI[x,y], (y, x))
		rpts = rpts.reshape((-1,1,2))
		descriptor.append((y, x, get_descriptor(I, x, y))) 
		#print rpts.shape, rpts
		cv2.polylines(I,[rpts],True,(0,1,0), 2)
		cv2.circle(I, (y, x), 1, (0,0,1), -1)	
	#cv2.putText(I, str(round(np.log10(threshold),3)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,1), 1, cv2.LINE_AA) #330
	cv2.imwrite(("hc_%.06f" % (threshold)) + imfile, I*255)
	#print len(descriptor), descriptor[0]
	return descriptor

'''
The function to compute sum of absolute distance between two feature descriptor vectors
'''
def sad_match(f1, f2, threshold=50):
	matches = {}
	for f in f1:
		(x1, y1, v1) = f
		mind, match = float('Inf'), None
		for h in f2:
			(x2, y2, v2) = h
			#print v1.shape, v2.shape
			if v1.shape != v2.shape: continue
			d = np.sum(abs(v1-v2)) 
			if d < mind: #and not (x2, y2) in matches.values():
				mind, match = d, (x2, y2)	
		matches[x1, y1] = match if mind < threshold else None #float('Inf')
		print (x1, y1), 'matches with', matches[x1, y1], 'with', mind
	return matches	
	
from matplotlib.patches import Polygon, ConnectionPatch
from matplotlib.collections import PatchCollection
'''
The main function to compute matches between two images using harris corner feature descriptors
Use compute_harris_corner() and sad_match functions to implement this function
'''
def compute_matches(imfile1, imfile2):
	
	f1 = HarrisCorner(imfile1, 0.000001)
	f2 = HarrisCorner(imfile2, 0.000001)
	fig = plt.figure(figsize=(10,5))
	print len(f1), len(f2)
	matches = sad_match(f1, f2)
	grad_kernel_x = np.array([[-1, 1]])		
	grad_kernel_y = np.array([[-1], [1]])
	
	I1 = cv2.imread(imfile1,0).astype(np.float) / 255.
	Ix, Iy = signal.convolve2d(I1, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(I1, grad_kernel_y, boundary='symm', mode='same')		
	mI, dI = np.sqrt(Ix**2 + Iy**2), np.arctan(Iy/(Ix+0.01))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	I1 = plt.imread(imfile1)
	ax1.imshow(I1); ax1.set_xticks([]); ax1.set_yticks([]) #, zorder=1)
	patches = []
	for (x,y,_) in f1:
		rpts = make_rectangle(25*mI[y,x], 25*mI[y,x], dI[y,x], (x, y))
		#print rpts
		polygon = Polygon(rpts, True)
		patches.append(polygon)
	p = PatchCollection(patches)#, alpha=0.1)
	ax1.add_collection(p)
	p.set_facecolor('none')
	p.set_edgecolor('red')
	p.set_linewidth(2)
	
	I2 = cv2.imread(imfile2,0).astype(np.float) / 255.
	Ix, Iy = signal.convolve2d(I2, grad_kernel_x, boundary='symm', mode='same'), signal.convolve2d(I2, grad_kernel_y, boundary='symm', mode='same')		
	mI, dI = np.sqrt(Ix**2 + Iy**2), np.arctan(Iy/(Ix+0.01))
	I2 = plt.imread(imfile2)
	ax2.imshow(I2); ax2.set_xticks([]); ax2.set_yticks([]) #, zorder=1)
	patches = []
	for (x,y,_) in f2:
		rpts = make_rectangle(25*mI[y,x], 25*mI[y,x], dI[y,x], (x, y))
		#print rpts
		polygon = Polygon(rpts, True)
		patches.append(polygon)
	p = PatchCollection(patches)#, alpha=0.1)
	ax2.add_collection(p)
	p.set_facecolor('none')
	p.set_edgecolor('red')
	p.set_linewidth(2)
	
	iter = 0
	for (x1, y1) in matches:
		print iter
		if matches[x1, y1] == None: continue
		(x2, y2) = matches[x1, y1]
		print (x1, y1), (x2, y2)
		con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",  axesA=ax2, axesB=ax1, color="red")
		ax2.add_artist(con)
		plt.tight_layout()
		plt.savefig("match_hc_" + str(iter).zfill(3) + '_' + imfile1)   # save the figure to file
		iter += 1

	#plt.scatter(10, 20, zorder=2)
	#plt.show()

# descriptor	
# https://www.youtube.com/watch?v=_qgKQGsuKeQ
	
'''	
#imfile = 'liberty1.jpg'  #'notredame1.jpg' #'yosemite1.jpg' #'me2.png' # 'me1.png' # 'chess.png' #'features_small.png' #'flower.jpg' #
#for th in np.linspace(0.000001, 0.0005, 25):
#	HarrisCorner(imfile, th)
imfile1, imfile2 = 'me.png', 'me3.png' #'liberty1.jpg', 'liberty2.jpg' #'horse1.png', 'horse2.png' #'mountain1.png', 'mountain2.png' #'yosemite1.jpg', 'yosemite2.jpg' #'trees_002.jpg', 'trees_003.jpg' # 'yard1.jpg', 'yard2.jpg' # 
compute_matches(imfile1, imfile2)
'''

#https://www.youtube.com/watch?v=O6E17BXIln4
from numpy.random import randint, choice
def rand_gen_sample(I, B):
	(m, n, d) = I.shape
	i = randint(0, m - B)
	j = randint(0, n - B)
	return I[i:i+B,j:j+B,:]	

def gen_samples(I, B, N):
	(m, n, d) = I.shape
	k = int(np.sqrt(N))
	print m, n, k, (m-B)/k, (n-B)/k
	samples = []
	i = 0
	while i <= m - B:
		j = 0
		while j <= n - B:
			samples.append(I[i:i+B,j:j+B,:])
			j += (n-B) / k
		i += (m-B) / k	
	return samples			

def ssd_patch(I1, I2, i, j, B, o):
	if i == 0 and j == 0:
		return 0
	else:
		ssd = 0
		if i > 0:
			ssd += np.sum((I1[i:i+o,j:j+B,:] - I2[:o,:B,:])**2)
		if j > 0:
			ssd += np.sum((I1[i:i+B,j:j+o,:] - I2[:B,:o,:])**2)
	return ssd			

def min_cut(p1, p2):
	(m, o, _) = p1.shape
	#e = np.zeros((m,o))	
	#for i in range(m):
	#	for j in range(1,o):
	#		e[i,j] = np.sum((p1[i,j-1,:] - p2[i,j,:])**2)
	e = np.sum((p1 - p2)**2, axis=2)
	inf = float('Inf')
	E = np.zeros((m,o))	
	j = 0
	for i in range(m):
		E[i,j] = e[i,j] 
	i = 0
	for j in range(o):
		E[i,j] = e[i,j] 
	for i in range(1,m):
		for j in range(1,o):
			E[i,j] = min([(E[i-1,j-1] if j > 1 else inf),E[i-1,j],(E[i-1,j+1] if j < o-1 else inf)]) + e[i,j]
	i = m - 1		
	_, minj = min([(E[i,j],j) for j in range(1,o)])
	indices = [minj]
	while i > 0:
		if minj > 1 and E[i-1,minj-1] + e[i,minj] == E[i,minj]:
			minj -= 1
		elif minj < o-1 and E[i-1,minj+1] + e[i,minj] == E[i,minj]:
			minj += 1
		indices = [minj] + indices
		i -= 1
	#print indices	
	return indices

def join(imfile1, imfile2, show_boundary=True):
	#I1 = cv2.imread(imfile1,3).astype(np.float) / 255. # 4-channel to 3-channel image 
	#cv2.imwrite(imfile1, I1)
	I2 = cv2.imread(imfile2,3).astype(np.float) #/ 255. # 4-channel to 3-channel image 
	cv2.imwrite(imfile2, I2)
	I1 = mpimg.imread(imfile1).astype(np.float) # * 255 # / 255
	I2 = mpimg.imread(imfile2).astype(np.float) #* 255 # / 255
	print I1.shape, I2.shape, np.max(I1), np.max(I2)
	imins = min_cut(np.transpose(I1,(1,0,2)),np.transpose(I2, (1,0,2)))
	jx = 0
	for imin in imins:
		I1[imin:,jx,:] = I2[imin:,jx,:] 
		if show_boundary:
			I1[imin,jx,:] = [1,0,0]
		jx += 1	
	plt.imshow(I1); plt.xticks([]); plt.yticks([]);
	plt.tight_layout()
	plt.savefig("join_" + imfile1)   # save the figure to file
	plt.show()
	plt.close()

def test_ssd(imfile, B, show_boundary=False):	
	o = B / 6
	I = mpimg.imread(imfile).astype(np.float) # * 255 # / 255
	(m, n, d) = I.shape
	print m, n, d
	samples = []
	i = j = 0
	while i <= m-B:
		j = 0
		while j <= n-B:
			samples.append(I[i:i+B,j:j+B])
			j += B - o
		i += B - o	
	print len(samples)	
	fig = plt.figure()
	fig.subplots_adjust(hspace=0.,wspace=0.) 	
	#plt.subplot_tool()
	for i in range(len(samples)):
		plt.subplot(6, 9,i+1);plt.imshow(samples[i]);plt.xticks([]);plt.yticks([])
	#plt.tight_layout()
	plt.savefig("samples_" + imfile)   # save the figure to file
	plt.show()
	plt.close()
	I1 = np.zeros((m, n, d))
	i, j = 0, 0
	while i < m:
		j = 0
		while j < n:
			if I1[i:i+B,j:j+B,:].shape == samples[0].shape:
				if i == 0 and j == 0:
					patches, index = [samples[0]], 0
				else:	
					ds = [ssd_patch(I1, I2, i, j, B, o) for I2 in samples]
					minindex = np.argmin(ds)
					mind = ds[minindex] 
					patches = [samples[minindex]]
					for ix in range(len(ds)):
						if abs(ds[ix] - mind) < 0.00001*mind and ds[ix] != mind:
							patches.append(samples[ix])
					# print mind		
					# choose one randomly
					index = np.random.choice(range(len(patches)), 1)[0]
				print len(patches), index
				I2 = patches[index]
				#print i, j, I1[i:i+B,j:j+B,:].shape, I2.shape
				if j > 0:
					jmins = min_cut(I1[i+o:i+B,j+B-o:j+B,:],I2[o:,B-o:B,:])
					ix = o
					for jmin in jmins:
						I1[i+ix,j+jmin:j+B,:] = I2[ix,jmin:B,:] 
						if show_boundary:
							I1[i+ix,j+jmin,:] = [1,0,0]
						ix += 1
				else:
					I1[i+o:i+B,j:j+B,:] = I2[o:B,:B,:] 
				if i > 0:		
					imins = min_cut(np.transpose(I1[i:i+o,j:j+B,:],(1,0,2)),np.transpose(I2[:o,:B,:], (1,0,2)))
					jx = 0
					for imin in imins:
						I1[i+imin:i+o,j+jx,:] = I2[imin:o,jx,:] 
						if show_boundary:
							I1[i+imin,j+jx,:] = [1,0,0]
						jx += 1	
				else:
					I1[i:i+o,j:j+B,:] = I2[:o,:B,:] 	
				#print len(imins), len(jmins)
				#I1[i:i+B,j:j+B,:] = I2 
			else:
				I1[i:i+B,j:j+B,:] = 0 
			j += B - o
		i += B	- o
	plt.imshow(I1[:i-(B-o),:j-(B-o)]); plt.xticks([]); plt.yticks([]);
	plt.tight_layout()
	plt.savefig("quilt_" + ('boundary_' if show_boundary else '') + imfile)   # save the figure to file
	plt.show()
	#plt.close()
		
	
def quilt(imfile, B, K=5, k=0.1, show_boundary=True):
	I = mpimg.imread(imfile).astype(np.float) # * 255 # / 255
	ns = 900 #625 #100
	samples = gen_samples(I, B, ns)
	print(len(samples))
	fig = plt.figure(figsize=(10,10))
	fig.subplots_adjust(hspace=0.,wspace=0.) 	
	ind = choice(range(len(samples)), 100)
	for i in range(100):
		plt.subplot(10, 10,i+1);plt.imshow(samples[ind[i]]);plt.xticks([]);plt.yticks([])
	#plt.tight_layout()
	plt.savefig("samples_" + imfile)   # save the figure to file
	plt.show()
	plt.close()
	o = B/6 #B / 5 #B/6
	(m, n, d) = I.shape
	M, N = (m/B)*B*K, (n/B)*B*K
	print B, m, n, M, N, d 	
	I1 = np.zeros((M, N, d))
	i, j = 0, 0
	niter = 0
	while i < M:
		j = 0
		while j < N:
			if I1[i:i+B,j:j+B,:].shape == samples[0].shape:
				ds = [ssd_patch(I1, I2, i, j, B, o) for I2 in samples]
				minindex = np.argmin(ds)
				mind = ds[minindex] 
				patches = [samples[minindex]]
				for ix in range(len(ds)):
					if abs(ds[ix] - mind) < k*mind and ds[ix] != mind:
						patches.append(samples[ix])
				# print mind		
				# choose one randomly
				index = np.random.choice(range(len(patches)), 1)[0]
				#print len(patches), index
				I2 = patches[index]
				#print i, j, I1[i:i+B,j:j+B,:].shape, I2.shape
				if j > 0:
					jmins = min_cut(I1[i+o:i+B,j+B-o:j+B,:],I2[o:,B-o:B,:])
					ix = o
					for jmin in jmins:
						I1[i+ix,j+jmin:j+B,:] = I2[ix,jmin:B,:] 
						if show_boundary:
							I1[i+ix,j+jmin,:] = [1,0,0]
						ix += 1
				else:
					I1[i+o:i+B,j:j+B,:] = I2[o:B,:B,:] 
				if i > 0:		
					imins = min_cut(np.transpose(I1[i:i+o,j:j+B,:],(1,0,2)),np.transpose(I2[:o,:B,:], (1,0,2)))
					jx = 0
					for imin in imins:
						I1[i+imin:i+o,j+jx,:] = I2[imin:o,jx,:] 
						if show_boundary:
							I1[i+imin,j+jx,:] = [1,0,0]
						jx += 1	
				else:
					I1[i:i+o,j:j+B,:] = I2[:o,:B,:] 	
				#print len(imins), len(jmins)
				#I1[i:i+B,j:j+B,:] = I2 
			else:
				I1[i:i+B,j:j+B,:] = 0 
			j += B - o
			if niter % 6 == 0:
				plt.imshow(I1[:M-(B-o),:N-(B-o)]); plt.xticks([]); plt.yticks([]);
				plt.tight_layout()
				plt.savefig("text_tran_" + ('boundary_' if show_boundary else '') + str(niter).zfill(4) + '_' + imfile)   # save the figure to file
				#plt.show()
				plt.close()		
			niter += 1
		i += B	- o
	plt.imshow(I1[:i-(B-o),:j-(B-o)]); plt.xticks([]); plt.yticks([]);
	plt.tight_layout()
	plt.savefig("quilt_" + ('boundary_' if show_boundary else '') + imfile)   # save the figure to file
	plt.show()
	plt.close()

def texture_transfer(imfile, imfile2, B, k=0.1, alpha=0.5, show_boundary=False):
	I = mpimg.imread(imfile).astype(np.float) #* 255 # / 255
	It = mpimg.imread(imfile2).astype(np.float) # * 255 # / 255
	ns = 900 #625 #100
	samples = gen_samples(I, B, ns)
	print(len(samples))
	'''	
	fig = plt.figure(figsize=(10,10))
	fig.subplots_adjust(hspace=0.,wspace=0.) 
	ind = choice(range(len(samples)), 100)
	for i in range(100):
		plt.subplot(10, 10,i+1);plt.imshow(samples[ind[i]]);plt.xticks([]);plt.yticks([])
	#plt.tight_layout()
	plt.savefig("samples_" + imfile)   # save the figure to file
	plt.show()
	plt.close()
	'''
	o = B / 6 #6 #4 #2 #B / 5 #B/6
	(m, n, d) = I.shape
	(M, N, _) = It.shape
	print B, m, n, M, N, d 	
	I1 = np.zeros((M, N, d))
	#niter = 5
	#for iter in range(niter):
	i, j = 0, 0
	niter = 0
	while i < M:
		j = 0
		while j < N:
			if I1[i:i+B,j:j+B,:].shape == samples[0].shape:
				ds = [alpha*ssd_patch(I1, I2, i, j, B, o) + (1-alpha)*np.sum((I2 - It[i:i+B,j:j+B,:])**2) for I2 in samples]
				minindex = np.argmin(ds)
				mind = ds[minindex] 
				patches = [samples[minindex]]
				for ix in range(len(ds)):
					if abs(ds[ix] - mind) < k*mind and ds[ix] != mind:
						patches.append(samples[ix])
				# print mind		
				# choose one randomly
				index = np.random.choice(range(len(patches)), 1)[0]
				#print len(patches), index
				I2 = patches[index]
				#print i, j, I1[i:i+B,j:j+B,:].shape, I2.shape
				if j > 0:
					jmins = min_cut(I1[i+o:i+B,j+B-o:j+B,:],I2[o:,B-o:B,:])
					ix = o
					for jmin in jmins:
						I1[i+ix,j+jmin:j+B,:] = I2[ix,jmin:B,:] 
						if show_boundary:
							I1[i+ix,j+jmin,:] = [1,0,0]
						ix += 1
				else:
					I1[i+o:i+B,j:j+B,:] = I2[o:B,:B,:] 
				if i > 0:		
					imins = min_cut(np.transpose(I1[i:i+o,j:j+B,:],(1,0,2)),np.transpose(I2[:o,:B,:], (1,0,2)))
					jx = 0
					for imin in imins:
						I1[i+imin:i+o,j+jx,:] = I2[imin:o,jx,:] 
						if show_boundary:
							I1[i+imin,j+jx,:] = [1,0,0]
						jx += 1	
				else:
					I1[i:i+o,j:j+B,:] = I2[:o,:B,:] 	
				#print len(imins), len(jmins)
				#I1[i:i+B,j:j+B,:] = I2 
			else:
				I1[i:i+B,j:j+B,:] = 0 
			'''
			if niter % 6 == 0:
				plt.imshow(I1[:M-(B-o),:N-(B-o)]); plt.xticks([]); plt.yticks([]);
				plt.tight_layout()
				plt.savefig("text_tran_" + ('boundary_' if show_boundary else '') + str(niter).zfill(4) + '_' + imfile)   # save the figure to file
				#plt.show()
				plt.close()		
			niter += 1
			'''
			j += B - o
		i += B	- o
	plt.imshow(I1[:i-(B-o),:j-(B-o)]); plt.xticks([]); plt.yticks([]);
	plt.tight_layout()
	plt.savefig("text_tran_" + ('boundary_' if show_boundary else '') + str(B).zfill(3) + '_' + imfile)   # save the figure to file
	plt.show()
	plt.close()	

def blend_with_mask(src, target, mask, alpha=0.5):
	Is = mpimg.imread(src).astype(np.float) #* 255 # / 255
	It = mpimg.imread(target).astype(np.float) # * 255 # / 255
	Im = mpimg.imread(mask).astype(np.float) / 255. # * 255 # / 255
	It = It[:Is.shape[0], :Is.shape[1],:]
	#Im = Im[:Is.shape[0], :Is.shape[1],:]
	print Is.shape, It.shape, Im.shape
	#Is *= Im
 	It[Im>0] = alpha*It[Im>0] + (1-alpha)*Is[Im>0] 
	plt.imshow(It); plt.xticks([]); plt.yticks([]);
	plt.tight_layout()
	plt.savefig("blend_" + target)   # save the figure to file
	plt.show()
	plt.close()	
	
imfile = 'stars.png' #'monet4.png' #'india.png' #'msky.png' # #'chiro.png' #'rabi2.png' #'r1.png' # #'notes.png' #'bangla.png' #'alphabet.png' #'rabi.png' #'epii.png' #'ds.png' #'q14.png' #'toast.png' #'q2.png' #'q14.png' #'toast.png' # # # #'cloud1.png' #'fire.png' # 'q14.png' #'q13.png' #'q10.png' #'q6.png' # #'q1.png' #'q11.png' #'q4.png' #'q9.png' #'q8.png' #'q5.png' #'q3.png' #'q7.png'	
B = 85 # #50 #250 #100 #50 #200 #25 #75
#test_ssd(imfile, B)	
#quilt(imfile, B, K=10, k=0.1, show_boundary=False)	
imfile2 = 'me6.png' #'janagana.png' #'mm.png' #'me6.png' # #'me6.png' #'me4.png' #'qt1.png' #'me.png' ##'mona.png' #'me1.png' # 
cv2.imwrite(imfile,cv2.imread(imfile,3).astype(np.float)) # * 255 # / 255
#for B in range(8, 3, -1):
texture_transfer(imfile, imfile2, B, k=0.1, alpha=0.5)
#join('i2.png', 'i1.png', False)
#cv2.imwrite('toast.png',cv2.imread('toast.png',3).astype(np.float)) # * 255 # / 255
#cv2.imwrite('text_tran_toast.png',cv2.imread('text_tran_toast.png',3).astype(np.float)) # * 255 # / 255
#cv2.imwrite('mask_toast.png',cv2.imread('mask_toast.png',3).astype(np.float)) # * 255 # / 255
#src, target, mask = 'text_tran_toast.png', 'toast.png', 'mask_toast.png' # 'me11.png'
#blend_with_mask(src, target, mask, 0.1)
#poisson_editing_color(src, target, mask)
#poisson_editing_color_mix(src, target, mask)

'''	
lines = read_file('inpros42.txt')
lines = read_file('inpros99.txt')
lines = read_file('rosalind_ba5b.txt')
n, m = map(int, str.split(lines[0]))
D = [[0 for _ in range(m + 1)] for _ in range(n)]
lines = lines[1:]
for i in range(n):
	D[i] = map(int, str.split(lines[i]))
R = [[0 for _ in range(m)] for _ in range(n + 1)]
lines = lines[n+1:]
for i in range(n + 1):
	R[i] = map(int, str.split(lines[i]))		
#print n, m
#print D
#print R
LongestPathManhattanTouristProblem(n, m, D, R)
'''

#pip install pip-autoremove
#pip-autoremove jupyter -y