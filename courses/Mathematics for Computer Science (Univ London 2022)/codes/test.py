def test():
	hex(int('d6af', 16) + int('91ff', 16))
	#float.hex(6.93)
	for A in range(28):
		if (181*A) % 29 == 1:
			print(A)
			break
	181*25 % 29

	for A in range(43):
		if (207*A) % 43 == 1:
			print(A)
			break

	207*16 % 43

import numpy as np
import matplotlib.pylab as plt

def create_image():
	image = np.zeros((4,4,3), dtype=np.uint8)
	image[0,0] = image[1,0] = image[0,3] = image[1,3] = [208,224,240]
	image[2,0] = image[2,3] = [112,160,64]
	image[3,0] = image[3,3] = [176,192,224]
	image[0,1] = image[1,1] = image[1,2] = [125,174,79]
	image[0,2] = [221,238,255]
	image[2,1] = [192,80,32]
	image[2,2] = [144,192,80]
	image[3,1] = [205,94,47]
	image[3,2] = [189,206,239]
	plt.imshow(image)
	plt.show()

from glob import glob
def read_images():
	images = []
	for f in sorted(glob('p*.txt')):
		image = np.zeros((4,4,3), dtype=np.uint8)
		with open(f) as f:
			lines = f.read().splitlines()
			i = 0
			while i < 4*3:
				for j in range(3):
					values = list(map(int, lines[i+j].split(' ')))
					for k in range(len(values)):
						image[i // 3, k, j] = values[k]
				i += 3
		#plt.imshow(image)
		#plt.show()
		images.append(image)
	return images
			
def extract_hidden_visualize(im):
	h, w, _ = im.shape
	m = np.zeros((h, w, 3), dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			for c in range(3):
				m[i,j,c] = (im[i,j,c] & 0x0F) << 4
			#print(m[i,j])
	#plt.imshow(m)
	#plt.show()
	return m

images = read_images()
letters = []
for image in images:
	letters.append(extract_hidden_visualize(image))

plt.figure(figsize=(15,18))
plt.subplots_adjust(0,0,1,1,0.05,0.05)
i = 1
for image in images:
	plt.subplot(3,4,i)
	plt.imshow(image)
	plt.axis('off')
	i += 1
#plt.tight_layout()
plt.show()

plt.figure(figsize=(15,18))
plt.subplots_adjust(0,0,1,1,0.05,0.05)
i = 1
for letter in letters:
	plt.subplot(3,4,i)
	plt.imshow(letter)
	plt.axis('off')
	i += 1
#plt.tight_layout()
plt.show()