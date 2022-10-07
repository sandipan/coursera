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
import cv2
import matplotlib.pylab as plt
from numpy import exp

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

from scipy import signal
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
	cv2.imwrite("tampered_" + imfile, f)	

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
# file:///C:/courses/Coursera/Past/Image%20Processing%20&%20CV/Duke%20-%20Image%20Processing/Week6/Perez03.pdf
import scipy

niter = 0
w, h = 0, 0

def callback(xk):
	global niter, w, h
	niter += 1
	print niter
	I = np.reshape(xk, (w,h))
	cv2.imwrite("pe_" + str(niter).zfill(3) + target, I*255)	

def poission_editing(src, target, mask):

	global w, h
	
	S = cv2.imread(src,0).astype(np.float) / 255
	T = cv2.imread(target,0).astype(np.float) / 255
	M =  cv2.imread(mask,0).astype(np.float) / 255
	(w, h) = S.shape
	#A = np.zeros((w*h, w*h), dtype='float32')
	#A = scipy.sparse.coo_matrix((w*h, w*h), dtype=np.int8) # empty matrix
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
	#A = scipy.sparse.csr_matrix(A)
	A = scipy.sparse.coo_matrix((data, (row, col)), shape=(w*h, w*h), dtype=np.int8) # empty matrix
	#I = np.reshape(scipy.sparse.linalg.cg(A, b, maxiter=200, callback=callback)[0], (w,h))
	#I = np.reshape(scipy.sparse.linalg.cgs(A, b, maxiter=200, callback=callback)[0], (w,h))
	I = np.reshape(scipy.sparse.linalg.spsolve(A, b), (w,h))
	cv2.imwrite("pe_" + target, I*255)	

col = None	
def callback2(xk): 
	global niter, w, h, col
	I = np.reshape(xk, (w,h))
	cv2.imwrite("pe_col_" + str(niter).zfill(3) + '_' + str(col) + target, I*255)	
	niter += 1
	print niter

import os 
	
def poission_editing_color(src, target, mask):

	global w, h, col, niter
	
	S = cv2.imread(src).astype(np.float) / 255
	T = cv2.imread(target).astype(np.float) / 255
	M =  cv2.imread(mask, 0).astype(np.float) / 255
	(w, h, d) = S.shape
	#A = np.zeros((w*h, w*h), dtype='float32')
	#A = scipy.sparse.coo_matrix((w*h, w*h), dtype=np.int8) # empty matrix
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
				#for dim in range(d):				
				#	b[h*x+y, dim] = s[dim] if abs(s[dim]) > abs(t[dim]) else t[dim]
			else:
				row.append(h*x+y)
				col.append(h*x+y)
				data.append(1)
				#A[h*x+y, h*x+y] = 1 
				b[h*x+y,:] = T[x,y,:]
			
	#I = np.reshape(np.linalg.solve(A, b), (w,h))
	#A = scipy.sparse.csr_matrix(A)
	A = scipy.sparse.coo_matrix((data, (row, col)), shape=(w*h, w*h), dtype=np.int8) # empty matrix
	#I = np.reshape(scipy.sparse.linalg.cg(A, b, maxiter=200, callback=callback)[0], (w,h))
	#I = np.reshape(scipy.sparse.linalg.cgs(A, b, maxiter=200, callback=callback)[0], (w,h))
	I = np.zeros((w, h, d))
	maxiter = 100
	for dim in range(d):
		#I[:,:,dim] = np.reshape(scipy.sparse.linalg.spsolve(A, b[:,dim]), (w,h))
		col = dim
		niter = 0
		I[:,:,dim] = np.reshape(scipy.sparse.linalg.cgs(A, b[:,dim], maxiter=maxiter, callback=callback2)[0], (w,h))
	cv2.imwrite("pe_" + target, I*255)		
	for i in range(maxiter):
		I = np.zeros((w, h, d))
		for dim in range(d):
			I[:,:,d-1-dim] = cv2.imread("pe_col_" + str(niter).zfill(3) + '_' + str(col) + target, 0)	
			os.remove("pe_col_" + str(niter).zfill(3) + '_' + str(col) + target)
		cv2.imwrite("pe_col_" + target, I*255)		

#http://onsmith.web.unc.edu/assignment-2/	
src = 'source.jpg' #'madurga.jpg' #'bear.bmp' #'peng.jpg'  #'liberty.jpg' ##'bird.jpg' #'walking3.jpg' #'madurga.jpg' # #
target = 'target.jpg' #'kol.jpg' #'waterpool.bmp' #'trekking.jpg' #'vic.jpg' # #'cloud.jpg' # 'water-small.jpg' #'kol.jpg' # #
mask = 'mask.jpg' #'maskp.jpg' #'mask.bmp' #'mask9.jpg' #'mask10.jpg' #'mask8.jpg' # #'mask7.jpg' #'maskp.jpg' # #
poission_editing_color(src, target, mask)	
#poission_editing(src, target, mask)	

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
	
#imfile = 'source.jpg' #'test.bmp'	
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

def aniso_diff(imfile): # heat
	#Parameters
	dt = 0.1;  #Time step
	T = 20;  #Stopping time
	K = 1;  #Conductivity
	a = 100 #5;
	u = cv2.imread(imfile,0).astype(np.float) / 255
	(m,n) = u.shape
	t = 0
	while t <= T:
		u1 = np.zeros((m+2, n+2))
		u1[1:1+m, 1:1+n] = u
		u_x = u1[2:2+m,1:1+n] - u
		u_y = u1[1:1+m,2:2+n] - u
		u_xx = u1[2:2+m,1:1+n] - 2*u + u1[0:m,1:1+n]
		u_yy = u1[1:1+m,2:2+n] - 2*u + u1[1:1+m,0:n]
		K = np.exp(-(u_xx+u_yy)**2/a)
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

def add_gaussian_noise(u, mean=0, std=0.1):
	return u + np.random.normal(mean, std, u.shape)
	
def TV_denoise(imfile, lambda1):
	#Parameters
	dt = 0.1;  #Time step
	T = 20;  #Stopping time
	u = cv2.imread(imfile,0).astype(np.float) #/ 255
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
		u += (u_xx*u_y**2 - 2*u_x*u_y*u_xy + u_yy*u_x**2) / (0.1 + (u_x**2 + u_y**2)**1.5)	- 2*lambda1*(u-f)	
		plt.imshow(u, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('TV denoise (t = ' + str(np.round(t,1)) + ')')
		plt.savefig("tv_denoise_" +  imfile.split('.')[0] + '_' + ("%05.1f" % t) + '.png')   # save the figure to file
		plt.close()
		t += dt

def TV_inpainting(imfile, mask, lambda1=0.2):
	
	#Parameters
	dt = 0.5;  #Time step
	T = 100;  #Stopping time
	D = cv2.imread(mask,0)
	D = D.astype(np.float) / 255
	f = cv2.imread(imfile,0)
	f = f.astype(np.float) / 255
	(m,n) = f.shape
	#R = np.random.random((m,n)) 
	#f = D * f + (1 - D) * R
	#print(D)
	#print(R)
	
	t = 0
	u = f
	uo = u
	ll = []
	while t <= T:
	
		u1 = np.zeros((m+2, n+2))
		u1[1:1+m, 1:1+n] = u
		u_x = u1[2:2+m,1:1+n] - u
		u_y = u1[1:1+m,2:2+n] - u
		u_xx = u1[2:2+m,1:1+n] - 2*u + u1[0:m,1:1+n]
		u_yy = u1[1:1+m,2:2+n] - 2*u + u1[1:1+m,0:n]
		u_xy = (u1[2:2+m,2:2+n] + u1[0:m,0:n] - u1[0:m,2:2+n] - u1[2:2+m,0:n]) / 4.
		u += (u_xx*u_y**2 - 2*u_x*u_y*u_xy + u_yy*u_x**2) / (0.01 + (u_x**2 + u_y**2)**1.5)	- 2*lambda1*D*(u-f)	
		
		l = np.linalg.norm(abs(uo - u), 'fro')
		ll.append(l)
		print(l)
		
		plt.imshow(u, cmap='gray')
		plt.title('TV impainting: Gradient Descent t = ' + str(t))
		plt.savefig("test/tv_inpaint_" +  '_' + ("%05.1f" % t) + imfile)   # save the figure to file
		plt.close()
		
		uo = u 
		t += dt

	#plt.imshow(np.abs(u-f), cmap='gray')
	plt.title('TV impainting: Gradient Descent t =  ' + str(t))
	plt.savefig("test/tv_inpaint_" +  '_' + ("%05.1f" % t) + imfile)   # save the figure to file
	plt.close()
	plt.plot(range(len(ll)), ll, '.-')
	plt.title('Gradient Descent Cost function decrease with iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Cost function')
	plt.savefig("test/tv_inpaint_gd_" + imfile)   # save the figure to file
	plt.close()		
		
#imfile = 'Cameraman256.png' #'lena.png'
#test_TV(imfile)	
#iso_diff(imfile)
#aniso_diff(imfile)
#u = cv2.imread(imfile,0).astype(np.float) / 255
#u = add_gaussian_noise(u)
#print u.shape
#cv2.imwrite('noisy_cam.png', u*255)
#imfile = 'noisy_cam.png' #'lena.png'
#lambda1 = 0.01 #1
#TV_denoise(imfile, lambda1)
#imfile, mask = 'tampered_lena1.png', 'mask1.png'
#TV_inpainting(imfile, mask)
 	
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