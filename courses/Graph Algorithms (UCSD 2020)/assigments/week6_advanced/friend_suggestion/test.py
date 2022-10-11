from glob import glob
from skimage.io import imread
import matplotlib.pylab as plt

dfiles = glob('out1d/*.png')
bdfiles = glob('out1bd/*.png')
afiles = glob('out1a/*.png')

id = 0
for i in range(max(len(dfiles), len(bdfiles), len(afiles))):
	if i < len(dfiles):
		imd = imread(dfiles[i])
	if i < len(bdfiles):
		imbd = imread(bdfiles[i])
	if i < len(afiles):
		ima = imread(afiles[i])
	plt.figure(figsize=(30,10))
	plt.subplots_adjust(0,0,1,1,0.01,0.01)
	plt.subplot(131), plt.imshow(imd), plt.axis('off')
	plt.subplot(132), plt.imshow(imbd), plt.axis('off')
	plt.subplot(133), plt.imshow(ima), plt.axis('off')
	plt.savefig('out/img_{:03d}.png'.format(id))
	plt.close()
	print(id)
	id += 1