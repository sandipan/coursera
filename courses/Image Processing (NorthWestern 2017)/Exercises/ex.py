import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import norm
#import seaborn as sns
import pandas as pd
import os
from math import log10

'''

## Week 2

img = cv2.imread(os.path.join(os.curdir, 'Week2\digital-images-week2_quizzes-lena.png'))
print img.shape

plt.subplot(421),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])

res = pd.DataFrame()
	
i = 2
for n in range(3, 16, 2):

	kernel = np.ones((n,n),np.float32)/(n*n)
	dst = cv2.filter2D(img,-1,kernel)
	#mse = (norm(img[:,:,1]-dst[:,:,1],'fro'))**2 / np.prod(img[:,:,1].shape)
	mse = np.sum((img[:,:,1]-dst[:,:,1])*(img[:,:,1]-dst[:,:,1])) / np.prod(img[:,:,1].shape)
	psnr = 10*log10(255**2/mse)
	res = res.append({'n': n, 'MSE': mse, 'PSNR': psnr}, ignore_index=True)

	plt.subplot(4,2,i),plt.imshow(dst),plt.title(str(n)+'x'+str(n)+' LPF')
	plt.xticks([]), plt.yticks([])
	i += 1

plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.tight_layout() 	
plt.show()

print res

plt.plot(res.n, res.MSE, '--r.')
plt.plot(res.n, res.PSNR, '--b.')
plt.xlabel('kernel width (n)')
plt.legend(),plt.title('Original Image and the Image after LPF')
plt.show()

'''

'''

## Week 3

img = cv2.imread('Week3/digital-images-week3_quizzes-original_quiz.jpg')
print img.shape

res = pd.DataFrame()

for n in range(3, 16, 2):

	print n

	fig = plt.figure()
	
	ax = plt.subplot2grid((6, 4), (0, 0), rowspan=2, colspan=2)
	ax.imshow(img),ax.set_title('Original (%02dx%02d)' % (img.shape[0], img.shape[1]),fontsize=8)
	ax.set_xticks([]), ax.set_yticks([])

	kernel = np.ones((n,n),np.float32)/(n*n)
	dst = cv2.filter2D(img,-1,kernel)

	ax1 = plt.subplot2grid((6, 4), (0, 2), rowspan=2, colspan=2)
	ax1.imshow(dst),ax1.set_title('after 3x3 LPF (%02dx%02d)' % (dst.shape[0], dst.shape[1]),fontsize=8)
	ax1.set_xticks([]), ax1.set_yticks([])

	dst = dst[1::2, 1::2]
	print dst.shape

	#plt.subplot(323),plt.imshow(dst, extent=[0, dst.shape[0], 0, dst.shape[1]], aspect='auto'),plt.title('downsampled (%02dx%02d)' % (dst.shape[0], dst.shape[1]))
	ax2 = plt.subplot2grid((6, 4), (2, 0))
	ax2.imshow(dst) #, origin='upper',extent=[dst.shape[0],2*dst.shape[0],2*dst.shape[1],dst.shape[1]], aspect='equal'),
	ax2.set_title('downsampled (%02dx%02d)' % (dst.shape[0], dst.shape[1]),fontsize=8)
	ax2.set_xticks([]), ax2.set_yticks([])

	dst1 = np.zeros(img.shape) #zeros(359,479);
	for i in range(dst1.shape[0]):
		for j in range(dst1.shape[1]):
			if i % 2 == 1 and j % 2 == 1:
				dst1[i, j] = dst[(i-1)/2, (j-1)/2]

	print dst1.shape

	ax3 = plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
	ax3.imshow(dst),ax3.set_title('upsampled (%02dx%02d)' % (dst1.shape[0], dst1.shape[1]),fontsize=8)
	ax3.set_xticks([]), ax3.set_yticks([])

	kernel = np.array([[0.25,0.5,0.25], [0.5,1,0.5], [0.25,0.5,0.25]])
	ax4 = plt.subplot2grid((6, 4), (4, 0))
	ax4.imshow(kernel),ax4.set_title('3x3 kernel',fontsize=8)
	ax4.set_xticks([]), ax4.set_yticks([])

	dst1 = cv2.filter2D(dst1,-1,kernel)
	
	ax5 = plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
	ax5.imshow(dst),ax5.set_title('upsampled and filtered (%02dx%02d)' % (dst1.shape[0], dst1.shape[1]),fontsize=8)
	ax5.set_xticks([]), ax5.set_yticks([])

	print dst1.shape

	mse = np.sum((img[:,:,1]-dst1[:,:,1])*(img[:,:,1]-dst1[:,:,1])) / np.prod(img[:,:,1].shape)
	psnr = 10*log10(255**2/mse)

	print mse, psnr	
	res = res.append({'n': n, 'MSE': mse, 'PSNR': psnr}, ignore_index=True)

	plt.subplots_adjust(left=0.1, right=0.6, bottom=0.05, top=0.95, wspace=0.1)
	#plt.tight_layout() 	

	fig.savefig('sampling' + str(n) + '.png')
	#plt.show()
	plt.close()

print res

plt.subplot(121),plt.plot(res.n, res.MSE, '--r.')
plt.xlabel('kernel width (n)')
plt.legend(),plt.title('Original Image and the Image after down/upsampling')
plt.subplot(122),plt.plot(res.n, res.PSNR, '--b.')
plt.xlabel('kernel width (n)')
plt.legend(),plt.title('Original Image and the Image after down/upsampling')
plt.show()

'''

'''
## Week 4

src_path = 'Week4/pichai1/' 
src = src_path + 'frame0000.png' 
left, top, right, bottom = 17, 442, 148, 536 #72,400,104,432 #64, 81, 96, 112

#src_path = 'Week4/motion_example/'
#src = src_path + 'frame0290.png' #'Week4/digital-images-week4_quizzes-frame_1.jpg'
#left, top, right, bottom = 72,400,104,432 #64, 81, 96, 112

#dst = src_path + 'frame0291.png' #'Week4/digital-images-week4_quizzes-frame_2.jpg'

bw, bh = right - left, bottom - top

imgsrc = cv2.imread(src)
imgsrc[left:right, bottom] = imgsrc[left:right, top] = 0
imgsrc[left, top:bottom] = imgsrc[right, top:bottom] = 0
#cv2.imwrite(src + '_with_block.jpg', imgsrc)

w = 5 # 10 #25
count = 1
for dirname, dirnames, filenames in os.walk(src_path):
	
	for filename in filenames:
	
		dst = dirname + filename
		print dst
		imgdst = cv2.imread(dst)
		
		B_target = imgsrc[left:right,top:bottom]
		#print np.mean(B_target, axis=2)
		M, N = imgdst.shape[0], imgdst.shape[1]
		min_mae = float('Inf')
		min_block = [-1, -1]
		for i in range(max(left-w,0), min(left+w,M-bw+1)): #range(M-bw+1):
			for j in range(max(top-w,0), min(top+w,N-bh+1)): #range(N-bh+1):
				B1 = imgdst[i:(i+bw),j:(j+bh)]
				mae = np.mean(abs(np.mean(B_target, axis=2) - np.mean(B1, axis=2))) #mae = np.mean(np.mean(abs(B_target - B1), axis=2))
				if mae < min_mae:
					min_mae = mae
					min_block = [i, j]

		print min_mae
		print min_block
		
		left = min_block[0]
		top = min_block[1]
		right, bottom = left + bw, top + bh
		
		imgdst[left:left+bw, top] = imgdst[left:left+bw, top+bh-1] = 0
		imgdst[left, top:(top+bh)] = imgdst[left+bw-1, top:(top+bh)] = 0
		cv2.imwrite(src_path + ('out/block_%03d' % (count)) + '.png', imgdst)
		
		count += 1
		imgsrc = imgdst
		
'''

'''
### Week 5

def filter_median(image, k):
  output = None
  # Insert your code here.----------------------------------------------------
  d = 2*k + 1

  output = np.zeros((image.shape[0] - d + 1, 
                     image.shape[1] - d + 1), dtype = image.dtype)
  
  for i in range(output.shape[0]):
    for j in range(output.shape[1]):
      patch = image[i:i+d, j:j+d]
      output[i,j] = np.median(patch)
  
  #---------------------------------------------------------------------------
  return output 

img_n = cv2.imread('Week5/digital-images-week5_quizzes-noisy.jpg')
img_o = cv2.imread('Week5/digital-images-week5_quizzes-original.jpg')
print img_o.shape

plt.subplot(331),plt.imshow(img_o),plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(img_n),plt.title('Noisy Image')
plt.xticks([]), plt.yticks([])

res = pd.DataFrame()

i = 3
for k in range(1,8):
	
	n = 2*k+1
	img_m = filter_median(img_n, k)
	#print img_m.shape
	#cv2.imwrite('digital-images-week5_quizzes-med-' + str(k) + '.png', img_m)
	
	mse = np.sum((img_o[k:-k,k:-k,1]-img_m)*(img_o[k:-k,k:-k,1]-img_m)) / np.prod(img_m.shape)
	psnr = 10*log10(255**2/mse)
	res = res.append({'n': n, 'MSE': mse, 'PSNR': psnr}, ignore_index=True)
	plt.subplot(3,3,i),plt.imshow(img_m, cmap='gray'),plt.title(str(n)+'x'+str(n)+' Median Filter')
	plt.xticks([]), plt.yticks([])
	i += 1

plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.tight_layout()	
plt.show()
	
print res

plt.subplot(121),plt.plot(res.n, res.MSE, '--r.')
plt.xlabel('median filter of width (n)')
plt.legend(),plt.title('Original Image and the Noisy Image after nxn Median Filter')
plt.subplot(122),plt.plot(res.n, res.PSNR, '--b.')
plt.xlabel('median filter of width (n)')
plt.legend(),plt.title('Original Image and the Noisy Image after nxn Median Filter')
plt.show()
'''

'''
### Week 6

## inverse filter with thresholding

## specify the threshold T
T = 0.5; #1e-1

## read in the original, sharp and noise-free image
original = np.mean(cv2.imread('Week6/original_cameraman.jpg'), axis=2) / 255
H, W = list(original.shape)

res = pd.DataFrame()

for T in np.linspace(0.1, 0.9, 9):
	
	print T
	
	fig = plt.figure(figsize=(10, 12))
		
	plt.subplot(431),plt.imshow(original, cmap='gray'),plt.title('original image', fontsize=8)
	plt.xticks([]), plt.yticks([])

	#print original

	## generate the blurred and noise-corrupted image for experiment
	motion_kernel = np.ones((1, 9)) / 9.  # 1-D motion blur
	motion_freq = np.fft.fft2(motion_kernel, s=(1024, 1024))  # frequency response of motion blur

	original_freq = np.fft.fft2(original, s=(1024, 1024))

	plt.subplot(432),plt.imshow(np.array(original_freq, dtype=float), cmap='gray'),plt.title('frequency response of original image', fontsize=8)
	plt.xticks([]), plt.yticks([])

	plt.subplot(433),plt.imshow(np.array(motion_freq, dtype=float), cmap='gray'),plt.title('frequency response of motion blur', fontsize=8)
	plt.xticks([]), plt.yticks([])

	blurred_freq = original_freq * motion_freq  # spectrum of blurred image

	plt.subplot(434),plt.imshow(np.array(blurred_freq, dtype=float), cmap='gray'),plt.title('spectrum of blurred image', fontsize=8)
	plt.xticks([]), plt.yticks([])

	blurred = np.fft.ifft2(blurred_freq)
	blurred = blurred[:H, :W]
	blurred[blurred < 0] = 0
	blurred[blurred > 1] = 1

	plt.subplot(435),plt.imshow(np.array(blurred, dtype=float), cmap='gray'),plt.title('blurred image', fontsize=8)
	plt.xticks([]), plt.yticks([])

	noisy = blurred + 0.2*blurred.std() * np.random.random(blurred.shape) #imnoise(blurred, 'gaussian', 0, 1e-4);
	#print noisy.shape, type(noisy)

	plt.subplot(436),plt.imshow(np.array(noisy, dtype=float), cmap='gray'),plt.title('noisy image', fontsize=8)
	plt.xticks([]), plt.yticks([])


	## Restoration from blurred and noise-corrupted image
	# generate restoration filter in the frequency domain
	inverse_freq = np.zeros(motion_freq.shape)
	inverse_freq[abs(motion_freq) < T] = 0
	inverse_freq[abs(motion_freq) >= T] = 1 / motion_freq[abs(motion_freq) >= T]

	#print inverse_freq.shape

	plt.subplot(437),plt.imshow(np.array(inverse_freq, dtype=float), cmap='gray'),plt.title('frequency response of restoration filter', fontsize=8)
	plt.xticks([]), plt.yticks([])

	# spectrum of blurred and noisy-corrupted image (the input to restoration)
	noisy_freq = np.fft.fft2(noisy, s=(1024, 1024))

	plt.subplot(438),plt.imshow(np.array(noisy_freq, dtype=float), cmap='gray'),plt.title('spectrum of blurred and noisy-corrupted image', fontsize=8)
	plt.xticks([]), plt.yticks([])

	# restoration
	restored_freq = noisy_freq * inverse_freq

	plt.subplot(439),plt.imshow(np.array(restored_freq, dtype=float), cmap='gray'),plt.title('spectrum of restored image', fontsize=8)
	plt.xticks([]), plt.yticks([])

	restored = np.fft.ifft2(restored_freq)
	restored = restored[:H, :W]
	restored[restored < 0] = 0
	restored[restored > 1] = 1

	plt.subplot(4,3,10),plt.imshow(np.array(restored, dtype=float), cmap='gray'),plt.title('restored image', fontsize=8)
	plt.xticks([]), plt.yticks([])

	plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.1, hspace=0.1)
	#plt.tight_layout() 	
	fig.savefig('inverse_filter_' + str(T) + '.png')
	#plt.show()
	plt.close()

	## analysis of result
	noisy_psnr = 10 * log10(1 / (norm(original - noisy, 'fro') ** 2 / H / W))
	restored_psnr = 10 * log10(1 / (norm(original - restored, 'fro') ** 2 / H / W))
	isnr = 10 * log10(norm(original - noisy, 'fro') ** 2 / (norm(original - restored, 'fro') ** 2))
	
	res = res.append({'T': T, 'NoisyPSNR': noisy_psnr, 'RestoredPSNR': restored_psnr, 'ISNR': isnr}, ignore_index=True)

	print noisy_psnr, restored_psnr, isnr

print res
	
plt.subplot(121),plt.plot(res.T, res.RestoredPSNR, '--r.')
plt.xlabel('T (threshold for inverse filtering)')
plt.legend(),plt.title('Restored PSNR with Inverse Filtering for different Thresholds')
plt.subplot(122),plt.plot(res.T, res.ISNR, '--b.')
plt.xlabel('T (threshold for inverse filtering)')
plt.legend(),plt.title('ISNR with Inverse Filtering for different Thresholds')
plt.show()

## visualization
#figure; imshow(original, 'border', 'tight');
#figure; imshow(blurred, 'border', 'tight');
#figure; imshow(noisy, 'border', 'tight');
#figure; imshow(restored, 'border', 'tight');
#figure; plot(abs(fftshift(motion_freq(1, :)))); title('spectrum of motion blur'); xlim([0 1024]);
#figure; plot(abs(fftshift(inverse_freq(1, :)))); title('spectrum of inverse filter'); xlim([0 1024]);
'''

'''
## Week 7
def next2pow(input):
	if input <= 0:
		print('Error: input must be positive!\n')
		result = -1
	else:
		index = 0
		while (1 << index) < input:
			index += 1
		result = 1 << index
	return result

def cls_restoration(image_noisy, psf, alpha):

	# find proper dimension for frequency-domain processing
	image_height, image_width = list(image_noisy.shape)
	psf_height, psf_width = list(psf.shape)
	dim = max([image_width, image_height, psf_width, psf_height]);
	dim = next2pow(dim);

%% frequency-domain representation of degradation
psf = padarray(psf, [dim - psf_height, dim - psf_width], 'post');
psf = circshift(psf, [-(psf_height - 1) / 2, -(psf_width - 1) / 2]);
H = fft2(psf, dim, dim);

%% frequency-domain representation of Laplace operator
Laplace = [0, -0.25, 0; -0.25, 1, -0.25; 0, -0.25, 0];
Laplace = padarray(Laplace, [dim - 3, dim - 3], 'post');
Laplace = circshift(Laplace, [-1, -1]);
C = fft2(Laplace, dim, dim);

%% Frequency response of the CLS filter
% Refer to the lecture for frequency response of CLS filter
% Complete the implementation of the CLS filter by uncommenting the
% following line and adding appropriate content
% R = conj(H) / (H.*conj(H) + alpha * (C.*conj(C)));
R = conj(H) ./ (abs(H).^2 + alpha * abs(C).^2);

%% CLS filtering
Y = fft2(image_noisy, dim, dim);
image_restored_frequency = R .* Y;
image_restored = ifft2(image_restored_frequency);
image_restored = image_restored(1 : image_height, 1 : image_width);

%% Simulate 1-D blur and noise
image_original = im2double(imread('Cameraman256.bmp', 'bmp'));
[H, W] = size(image_original);
blur_impulse = fspecial('motion', 7, 0);
image_blurred = imfilter(image_original, blur_impulse, 'conv', 'circular');
noise_power = 1e-4;
randn('seed', 1);
noise = sqrt(noise_power) * randn(H, W);
image_noisy = image_blurred + noise;

figure; imshow(image_original, 'border', 'tight');
figure; imshow(image_blurred, 'border', 'tight');
figure; imshow(image_noisy, 'border', 'tight');

%% CLS restoration
alpha = 0.1; %1;  % you should try different values of alpha
image_cls_restored = cls_restoration(image_noisy, blur_impulse, alpha);
figure; imshow(image_cls_restored, 'border', 'tight');

%% computation of ISNR
% ...
isnr = 10 * log10(norm(image_original - image_noisy, 'fro') ^ 2 / (norm(image_original - image_cls_restored, 'fro') ^ 2));
isnr
'''

'''
## Watermarking
cover = np.mean(cv2.imread('Week7/Cameraman512.png'), axis=2) # cover size = 8 x  watermark size # Cameraman_bright
#print cover.shape
watermark = np.mean(cv2.imread('Week7/watermark.png'), axis=2).astype('uint8') #np.unpackbits(np.mean(cv2.imread('Week7/watermark.png'), axis=2).astype('uint8')) 
#print len(watermark)
#cover_mask = np.ones(8) # np.unpackbits(np.array([(2**bit)^0xFF], dtype='uint8'))
for bit in range(8): # 0..7
	water_mask = 1 << bit #2**bit
	cover_mask = (water_mask)^0xFF #np.uint8((2**bit)^0xFF)
	#cover = cover.T
	for i in range(watermark.shape[0]):
		for j in range(watermark.shape[1]): #, cover.shape[1]):
			#cover[i,j] = min(cover[i,j] + 100, 255) 
			cover[i,j] = (int(cover[i,j]) & cover_mask) + (watermark[i,j] & water_mask)
	#cover = cover.T
	#cv2.imwrite('Cameraman_bright.png', cover)
	cv2.imwrite('watermarked_' + str(bit) + '.png', cover)
	plt.subplot(3,3,bit+1),plt.hist(cover.flatten(), 50, normed=1, facecolor='green', alpha=0.75),plt.title('Embedding in the bit plane ' + str(bit), fontsize=8)
	plt.xlabel('Pixel values')
	plt.ylabel('Distribution')
	plt.grid()

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout() 	
#fig.savefig('inverse_filter_' + str(T) + '.png')
plt.show()
'''

'''
from PIL import Image, ImageDraw
main = Image.open("Week7/Cameraman512.png")
watermark = Image.new("RGBA", main.size)
waterdraw = ImageDraw.ImageDraw(watermark, "RGBA")
waterdraw.text((10, 10), "Sandipan Dey")
watermask = watermark.convert("L").point(lambda x: min(x, 100))
watermark.putalpha(watermask)
main.paste(watermark, None, watermark)
main.save("Cameraman512_watermarked.png", "PNG")

#from PIL import Image
#photo = Image.open("Week7/Cameraman512.png")
#watermark = Image.open("Week7/watermark.png")
#photo.paste(watermark, (0, 0), watermark)
#photo.save("watermarked_image.png")

'''

'''
## Week 8
def get_distr(I):
	
	m, n = list(I.shape)
	probs = np.zeros(256)
	for i in range(m):
		for j in range(n):
			probs[I[i,j]] += 1.
	probs = probs / (m*n)
	return probs
	
def hist_equalize(I, probs):
	
	T = map(int, 255*np.cumsum(probs))
	m, n = list(I.shape)
	for i in range(m):
		for j in range(n):
			I[i,j] = T[I[i,j]]
	return I

from heapq import *
from graphviz import Digraph
import csv

def huffman_encoding(probs, path, save_graph=True):
	
	h = []
	for i in range(256):
		if probs[i] > 0:
			heappush(h, (probs[i], str(i)))
	
	s = None
	
	if save_graph:
		s = Digraph('htree', node_attr={'shape': 'plaintext'}, format='png') #, engine='neato')
		s.attr('node', fontcolor='gray')
	
	k = 1
	
	par = {}
	while len(h) > 0:
		
		pr1, pi1 = heappop(h)
		if len(h) == 0:
			break
		pr2, pi2 = heappop(h)
		if save_graph:
			s.node(pi1, str(round(pr1,8)) + ('' if '_' in pi1 else '(' + pi1 + ')'))
			s.node(pi2, str(round(pr2,8)) + ('' if '_' in pi2 else '(' + pi2 + ')'))
			s.node(pi1+'_'+pi2, str(round(pr1+pr2,8)))
			s.edge(pi1+'_'+pi2, pi1, label='0')
			s.edge(pi1+'_'+pi2, pi2, label='1')
			s.render(path + 'htree_' + str(k), view=False)
		heappush(h, (pr1+pr2,pi1+'_'+pi2))
		par[pi1], par[pi2] = (pi1+'_'+pi2, '0'), (pi1+'_'+pi2, '1')
		k += 1
	
	#colors = plt.cm.BuPu(np.linspace(0, 0.5, 20))
	#columns = ('pixel', 'code')
	#cols = []
	dict = {}
	#cell_text = []
	for pix in set(par.keys()): 	
		
		bits = ''
		if '_' in pix:
			continue
		p = pix
		while p in par:
			p, bit = par[p]
			bits += bit
		#print pix, bits
		dict[pix] = bits
		#cols.append(colors[len(bits)])
		#cell_text.append([pix, bits])
	
	with open(path+ 'codes.csv', 'wb') as f:  
		w = csv.writer(f)
		w.writerows(dict.items())

	#plt.table(cellText=cell_text,
	#			#animated=True,
	#			rowLabels=None,
	#			colColours=cols,
	#			colLabels=columns,
	#			loc='center')
	#plt.xticks([]), plt.yticks([])
	#plt.show()

	#print dict
	
	return dict

def encode_huffman(I, dict):
	
	isize = np.prod(I.shape)
	bits = ''
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			bits += dict[str(I[i,j])]
	#print bits
	csize = len(bits) / 8
	return isize, csize, isize / (1.*csize)

def LZ78(I):

	dict = {str(i):"{0:b}".format(i) for i in range(256)}
	isize = np.prod(I.shape)
	bits = ''
	pixels = I.flatten()
	i = 0
	while i < len(pixels):
		symbol = str(pixels[i])
		i += 1
		while i < len(pixels) and symbol + ',' + str(pixels[i]) in dict:
			symbol +=  ',' + str(pixels[i])
			i += 1
		bit = dict[symbol]	
		if i < len(pixels):
			dict[symbol + ',' + str(pixels[i])] = bit + dict[str(pixels[i])]
			bit += dict[str(pixels[i])]
		i += 1
		bits += bit
		
	with open('codes_LZ78.csv', 'wb') as f:  
		w = csv.writer(f)
		w.writerows(dict.items())
		
	csize = len(bits) / 8
	asize = [len(x) in dict.values()]
	return isize, csize, isize / (1.*csize)
		
def entropy(probs):
	
	#probs[probs == 0] = 10**(-20)
	probs = probs[probs > 0]
	return -np.sum(probs * (np.log2(probs)))

image = 'beans.png' #'girl.png' #'monument.png' #'beans.png' #'girl.png' #Cameraman256.bmp
I = np.mean(cv2.imread('Week8/'+image), axis=2).astype('uint8') 
#isz, cszl, compl = LZ78(I)
probs = get_distr(I)
I = hist_equalize(I, probs)
isz, cszl, compl = LZ78(I)
'''

'''
df_o = pd.DataFrame()
df_e = pd.DataFrame()
#image = 'girl.png' #'monument.png' #'beans.png' #'girl.png' #Cameraman256.bmp
for image in ['girl.png', 'beans.png', 'monument.png', 'lena.png', 'fruit.png', 'Cameraman256.png', 'peacock.png', 'rose.png', 'fractal.png']:
	
	print image
	I = np.mean(cv2.imread('Week8/'+image), axis=2).astype('uint8') 
	probs = get_distr(I)
	
	#plt.subplot(121),plt.hist(I.flatten(), 50, normed=1, facecolor='green', alpha=0.75), plt.title('Probability Distribution')
	#plt.subplot(122),plt.plot(np.cumsum(probs), '-.r'), plt.title('Cumulative Probability Distribution')
	#plt.show()
	
	#print probs
	#print sum(probs)
	
	e = entropy(probs)
	dict = huffman_encoding(probs, 'Week8/htree/', False)
	isz, cszh, comph = encode_huffman(I, dict)
	isz, cszl, compl = LZ78(I)
	df_o = df_o.append({'image': image.split('.')[0], 'size':str(I.shape), 'entropy': e, 'datasize': isz, 'Hcompresseddatasize': cszh, 'Hcompressionratio': comph, 'Lcompresseddatasize': cszl, 'Lcompressionratio': compl}, ignore_index=True)
	
	I = hist_equalize(I, probs)
	probs = get_distr(I)
	
	#plt.subplot(121),plt.hist(I.flatten(), 50, normed=1, facecolor='green', alpha=0.75), plt.title('Probability Distribution')
	#plt.subplot(122),plt.plot(np.cumsum(probs), '-.r'), plt.title('Cumulative Probability Distribution')
	#plt.show()
	cv2.imwrite('equalized_' + image, I)
	
	#print probs
	#print sum(probs)
	e = entropy(probs)
	dict = huffman_encoding(probs, 'Week8/htree2/', False)
	isz, cszh, comph = encode_huffman(I, dict)
	isz, cszl, compl = LZ78(I)
	df_e = df_e.append({'image': image.split('.')[0], 'size':str(I.shape), 'entropy': e, 'datasize': isz, 'Hcompresseddatasize': cszh, 'Hcompressionratio': comph, 'Lcompresseddatasize': cszl, 'Lcompressionratio': compl}, ignore_index=True)

df_o.to_csv('original_images.csv')
df_e.to_csv('equalized_images.csv')

#plt.subplot(121), 
df_o.plot(x='image', y=['datasize', 'Hcompresseddatasize', 'Lcompresseddatasize'], kind='bar')
plt.show()
df_o.plot(x='image', y=['entropy', 'Hcompressionratio', 'Lcompressionratio'], kind='line',  rot=90)
plt.show()
df_e.plot(x='image', y=['datasize', 'Hcompresseddatasize', 'Lcompresseddatasize'], kind='bar')
plt.show()
df_e.plot(x='image', y=['entropy', 'Hcompressionratio', 'Lcompressionratio'], kind='line',  rot=90)
plt.show()

'''

'''
def ostu(probs, image, draw=True):
	P1 = np.cumsum(probs)
	P2 = 1. - P1
	m = np.zeros(256)
	for k in range(256):
		m[k] = (0. if k == 0 else m[k-1]) + k*probs[k]
	mG = m[255] #sum([i*probs[i] for i in range(256)])
	vars, varmax, kmax = [], -1, -1
	for k in range(256):
		var = (mG*P1[k]-m[k])**2 / (P1[k]*P2[k])
		vars.append(var)
		if var > varmax:
			varmax, kmax = var, k
	#print varmax, kmax
	if draw:
		plt.bar(range(256), vars, color='green',width=1.0), plt.xlabel('pixel value'), plt.ylabel('variance'), plt.title('Optimal thresholding with Ostu\'s method for the image ' + image.split('.')[0])
		plt.axvline(x=kmax, color='k', label = 'optimal threshold = ' + str(kmax))
		plt.legend()
		plt.show()
	return kmax

def threshold(I, k):
	I[I < k] = 0
	I[I >= k] = 255
	return I

def get_distr(I):
	
	m, n = list(I.shape)
	probs = np.zeros(256)
	for i in range(m):
		for j in range(n):
			probs[I[i,j]] += 1.
	probs = probs / (m*n)
	return probs
'''

'''
for image in ['girl.png', 'monument.png', 'beans.png', 'Cameraman256.png', 'lena.png', 'equalized_girl.png', 'equalized_beans.png', 'equalized_lena.png', 'equalized_monument.png']:
	I = np.mean(cv2.imread('Week8/'+image), axis=2).astype('uint8') 
	probs = get_distr(I)
	k = ostu(probs, image)
	I = threshold(I, k)
	cv2.imwrite('binarized_'+image, I)
'''

'''
def sobel_filter_x():
  return np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

def sobel_filter_y():
  return np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])

from numpy import cos, sin, pi
def hough_transform(I, threshold=10):
	#epsilon = 1 #1e-1
	#rhos = np.linspace(-200,200,400)
	thetas = np.linspace(-pi/2,pi/2,360)
	#rG, tG = np.meshgrid(rhos, thetas) # create the actual grid
    #rG = rG.flatten() # make the grid 1d
    #tG = tG.flatten() # same
    #hdf = pd.DataFrame({'rho':rG, 'theta':tG})
	hdict = {}
	count  = 1
	vote_points = {}
	#rhos = set([])
	for x in range(I.shape[0]):
		for y in range(I.shape[1]):
			if I[x,y] < 255: continue 
			print count
			count += 1
			#for rho in rhos:
			for theta in thetas:
				#if abs(x*cos(theta) + y*sin(theta) - rho) < epsilon:
				rho = round(x*cos(theta) + y*sin(theta))
				#rhos.add(rho)
				hdict[rho,theta] = hdict.get((rho,theta),0) + 1
				vote_points[rho, theta] = vote_points.get((rho, theta), []) + [(x,y)]
	#rhos = list(rhos)
	rho_theta = map(list, zip(*hdict.keys()))
	#print len(rho_theta[0]), len(rho_theta[1]), len(hdict.values()) 
	hdf = pd.DataFrame({'rho' : rho_theta[0] , 'theta' : rho_theta[1], 'votes': hdict.values()})
	hdf = hdf.pivot(index='rho', columns='theta', values='votes') #.as_matrix())
	hdf.to_csv('hog.csv')
	#print(min(rhos), max(rhos))
	plt.imshow(hdf, cmap=plt.cm.YlOrRd) #plt.cm.RdPu)
	plt.colorbar()
	plt.xticks([0, 120, 180, 240, 360], [r'$-\frac{\pi}{2}$',r'$-\frac{\pi}{4}$',r'$0$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$']), plt.yticks(np.linspace(-100, 1000, 9)), plt.xlabel(r'$\theta$'), plt.ylabel(r'$\rho$') 
	plt.title(r'Hough Transofrm $\rho-\theta$ space')
	#plt.yticks(np.linspace(min(rhos), max(rhos), 5))
	plt.show()
	rhos = list(hdf.index)
	hdf = np.nan_to_num(hdf.as_matrix())
	#rho_ind, theta_ind = np.unravel_index(hdf.argmax(), hdf.shape)
	#rho, theta = rhos[rho_ind], thetas[theta_ind]
	#print rho, theta, hdf.max()
	plt.imshow(I, cmap='gray') 
	for rho_ind, theta_ind in zip(*np.where(hdf > threshold)):
		rho, theta = rhos[rho_ind], thetas[theta_ind]
		print rho, theta
		x, y = zip(*vote_points[rho, theta])
		plt.scatter(y, x, c='red', s=1)
		#plt.plot(range(I.shape[0]), (rho - np.array(range(I.shape[0]))*cos(theta)) / sin(theta), color='red')
	#plt.ylim([0, I.shape[1]])
	plt.show()

from scipy.signal import convolve2d

#I = np.mean(cv2.imread('rect.png'), axis=2).astype('uint8')  # 'rect.png' 'dotline.png'
#hough_transform(I)
image = 'umbc.png' #'tiger.png'
I = np.mean(cv2.imread(image), axis=2).astype('uint8')  # 'rect.png' 'dotline.png'
I = convolve2d(I, sobel_filter_y()) #sobel_filter_x())
plt.imshow(I, cmap='gray')
plt.show()
I[I < 0] = 0
I[I > 255] = 255
plt.imshow(I, cmap='gray')
plt.show()
probs = get_distr(I)
k = ostu(probs, image)
I = threshold(I, k)
cv2.imwrite('binarized_'+image, I)
I = np.mean(cv2.imread('binarized_umbc.png'), axis=2).astype('uint8')  # 'rect.png' 'dotline.png' #'binarized_tiger.png'
hough_transform(I, 90) #95 90 80

#How Hough Transform works Thales Sehn Korting	https://www.youtube.com/watch?v=4zHbI-fFIlI
'''

'''
## Week 9
I = imread('Cameraman256.bmp');
imshow(double(I) / 255);
imwrite(I, 'Cameraman256.jpg', 'jpg', 'quality', 10);

I2 = imread('Cameraman256.jpg');
I2 = double(I2);
I = double(I);
N1 = size(I2,1);
N2 = size(I2,2);
MSE = 0;
for i=1:N1
    for j=1:N2
        MSE = MSE + (I2(i, j) - I(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
MSE = (norm(I2-I, 'fro'))^2 / (N1 * N2);
PSNR = 10*log10(255^2/MSE);

'''

## Week 11
'''
A = np.zeros((256,256)) # initialize a 256*256 image  

# initialize absolute ADI, positive ADI and Negative ADI
# all initialized to zero
# Note that all ADIs are of the same size with the image
# DO NOT change the name of the ADIs as they will be used later
ADI_abs = np.zeros((256,256))
ADI_pos = np.zeros((256,256))
ADI_neg = np.zeros((256,256))

# initialize the starting position of the moving object
# the moving object is a rectangle similar to the example in the lecture
# slides
start1 = 100;
start2 = 150;
start3 = 40;
start4 = 110;

#threshold T as in equations in the lecture slides regarding ADI
T = 0.5;

#initialize the reference frame R
A[start1:start2, start3:start4] = 1

cv2.imwrite('Week11/frames/motion0.png', A*255)

#visualize the object and in the reference frame R
plt.imshow(A, cmap='gray')
plt.show()

j = 0
for i in range(5, 55, 5):
        j += 12;
        A2 = np.zeros((256,256));
        A2[start1 + i: start2 + i, start3 + j: start4 + j] = 1;
        cv2.imwrite('Week11/frames/motion'+str(i)+'.png', A2*255)
        
        # You need to code up the follwing part that calculate the ADIs
        # Namely, the absolute ADI, the positive ADI and the negative ADI
        # Equations can be found in lecture slides regarding ADIs
        # You need to decide on the appropriate threshold T for this case
        # at line 23
        for x in range(256):
            for y in range(256):
                if abs(A[x,y] - A2[x,y]) > T:
                    ADI_abs[x,y] = ADI_abs[x,y] + 1; 
                if (A[x,y] - A2[x,y]) > T:
                    ADI_pos[x,y] = ADI_pos[x,y] + 1;
                if (A[x,y] - A2[x,y]) < -T:
                    ADI_neg[x,y] = ADI_neg[x,y] + 1;

# The following part will calculate the moving speed 
# and the total space(in pixel number) occupied by the moving object
#[row, col] = np.where(ADI_neg > 0);
#speed_X_Direction = (max(col) - start4) / 10
#speed_Y_Direction = (max(row) - start2) / 10
#total_space_occupied = sum(sum(ADI_abs > 0))

# The following part helps you to visualize the ADIs you compute
# compare them with the example shown in lecture
# You should be getting someting very similar
plt.imshow(ADI_abs,cmap='gray')
plt.show()
plt.imshow(ADI_pos,cmap='gray')
plt.show()
plt.imshow(ADI_neg,cmap='gray')
plt.show()
'''

'''
for dirname, dirnames, filenames in os.walk('Week4/kalmrigaya/'):
	
	for filename in filenames:
		I = np.mean(cv2.imread('Week4/kalmrigaya/' + filename), axis=2).astype('uint8') 
		probs = get_distr(I)
		k = ostu(probs, None, False)
		I = threshold(I, k)
		cv2.imwrite('Week4/kalmrigaya/bin/binarized_'+filename, I)
'''

'''
T = 100 #25 #100 #100

#A = np.mean(cv2.imread('Week4/bin/binarized_frame0290.png'), axis=2).astype('uint8') 
#A = np.mean(cv2.imread('Week4/sachin/bin/binarized_frame0040.png'), axis=2).astype('uint8') 
#A = np.mean(cv2.imread('Week4/football/bin/binarized_frame0024.png'), axis=2).astype('uint8') 
A = np.mean(cv2.imread('Week4/kalmrigaya/bin/binarized_frame0351.png'), axis=2).astype('uint8') 
#ADI_abs = np.zeros(A.shape)
#ADI_pos = np.zeros(A.shape)
#ADI_neg = np.zeros(A.shape)

for dirname, dirnames, filenames in os.walk('Week4/kalmrigaya//bin/'): #'Week4/bin/'
    
    for filename in filenames:
        print filename
        ADI_abs = np.zeros(A.shape)
        #ADI_pos = np.zeros(A.shape)
        #ADI_neg = np.zeros(A.shape)
        A2 = np.mean(cv2.imread('Week4/kalmrigaya/bin/' + filename), axis=2).astype('uint8') #Week4/bin/
        for x in range(A.shape[0]):
            for y in range(A.shape[1]):
                if abs(A[x,y] - A2[x,y]) > T:
                    ADI_abs[x,y] = ADI_abs[x,y] + 255; 
                #if (A[x,y] - A2[x,y]) > T:
                #   ADI_pos[x,y] = ADI_pos[x,y] + 255;
                #if (A[x,y] - A2[x,y]) < -T:
                #    ADI_neg[x,y] = ADI_neg[x,y] + 255;
        #A3 = np.mean(cv2.imread('Week4/kalmrigaya/' + filename.split('_')[1]), axis=2).astype('uint8') #Week4/bin/
        #ADI_abs += A3
        #ADI_pos += A3
        #ADI_neg += A3
        A = A2
        cv2.imwrite('Week4/kalmrigaya/adi/adi_abs_'+filename, ADI_abs) #'Week4/adi/adi_abs_'
        #cv2.imwrite('Week4/kalmrigaya/adip/adi_pos_'+filename, ADI_abs) #'Week4/adi/adi_abs_'
        #cv2.imwrite('Week4/kalmrigaya/adin/adi_neg_'+filename, ADI_abs) #'Week4/adi/adi_abs_'

plt.imshow(ADI_abs,cmap='gray')
plt.show()
#plt.imshow(ADI_pos,cmap='gray')
#plt.show()
#plt.imshow(ADI_neg,cmap='gray')
#plt.show()
'''

'''
## Week 12
D = np.array([[np.sin(i+j) for i in range(10)] for j in range(10)])
A = D + np.eye(10)
b = [-2,-6,-9,1,8,10,1,-9,-4,-3]

from sklearn.linear_model import OrthogonalMatchingPursuit
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3)
omp.fit(A, b)
coef = omp.coef_
print coef
idx_r, = coef.nonzero()
print idx_r
#print dir(omp)
plt.subplot(4, 1, 2)
plt.xlim(-1, 10)
plt.title("Recovered signal from noise-free measurements")
plt.stem(idx_r, coef[idx_r])
plt.show()

for i in range(A.shape[1]):
    A[:,i] = A[:,i] / np.sqrt(np.sum(A[:,i]**2))
#print np.dot(A[:,1].T, A[:,1]) #sum(A[:,1]**2)

#b = np.reshape(b, (10,1))
S = 3
All = set(range(10))
Omega = set([])
r = b

while S > 0:
    x = np.zeros(10)
    x_i, i = -1, -1
    for j in All-Omega:
        x_j = abs(np.dot(A[:,j], r)) #sum(A[:,j]*r))
        print j, x_j
        if x_j > x_i:
            x_i, i = x_j, j
    #print i
    Omega = Omega.union([i])
    #print Omega, All - Omega
    A1 = A[:,list(Omega)]
    x_Omega_star = np.matmul(np.linalg.inv(np.matmul(A1.T,A1)), np.matmul(A1.T,b)) # never use '*' for matrix multiplication
    r = b - np.matmul(A1, x_Omega_star) 
    x[list(Omega)] = x_Omega_star 
    print x, Omega
    S -= 1

#print np.sum((np.matmul(A,x)-b)**2), np.sum((np.matmul(A,coef)-b)**2), 
'''