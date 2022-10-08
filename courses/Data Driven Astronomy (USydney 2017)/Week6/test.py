import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import statistics
import time
import sys

# Write your calc_stats function here.
def calc_stats(f):
  x = np.loadtxt(f,delimiter=',')
  return (round(np.mean(x),1),round(np.median(x),1))

def mean_datasets(flist):
  x = np.loadtxt(flist[0],delimiter=',')
  for f in flist[1:]:
	x += np.loadtxt(f,delimiter=',')
  return np.round(x / len(flist), 1)  


def load_fits(f):
	hdulist = fits.open(f)
	data = hdulist[0].data
	return np.unravel_index(data.argmax(), data.shape)
	# Plot the 2D array
	#plt.imshow(data, cmap=plt.cm.viridis)
	#plt.xlabel('x-pixels (RA)')
	#plt.ylabel('y-pixels (Dec)')
	#plt.colorbar()
	#plt.show()

def mean_fits(flist):
  x = fits.open(flist[0])[0].data
  for f in flist[1:]:
	x += fits.open(f)[0].data
  return x / len(flist)  

def list_stats(x):
  n = len(x)
  x = sorted(x)
  return (x[n//2] if n%2 == 1 else (x[n//2-1]+x[n//2])/2., sum(x) / float(n))

def time_stat(func, size, ntrials):
  # the time to generate the random array should not be included
  total = 0
  for i in range(ntrials):
    data = np.random.rand(size)
    # modify this function to time func with ntrials times using a new random array each time
    start = time.perf_counter()
    res = func(data)
    total += time.perf_counter() - start
  # return the average run time
  return total / float(ntrials)

#a = np.array([])
#b = np.array([1, 2, 3])
#c = np.zeros(10**6)
#for obj in [a, b, c]:
#  print('sys:', sys.getsizeof(obj), 'np:', obj.nbytes)
#a = np.zeros(5, dtype=np.int32)
#b = np.zeros(5, dtype=np.float64)
#for obj in [a, b]:
#  print('nbytes         :', obj.nbytes)
#  print('size x itemsize:', obj.size*obj.itemsize)

def median_fits(flist):
  x = fits.open(flist[0])[0].data
  N, m, n = len(flist), x.shape[0], x.shape[1]
  start = time.perf_counter()
  ax = np.zeros((N, m, n))#, dtype=np.float32)
  for i in range(N):
    ax[i,:,:] = fits.open(flist[i])[0].data
  x = np.zeros((m, n))
  #for i in range(m):
  #  for j in range(n):
  #      x[i,j] = np.median(ax[:,i,j])
  x = np.median(ax, axis=0)
  seconds = time.perf_counter() - start
  print(sys.getsizeof(ax) / 1024., ax.nbytes / 1024., N*m*n*8 / 1024.) #, N*m*n*4 / 1024.)
  return (x, seconds, sys.getsizeof(ax) / 1024.)

def median_bins(values, B):
	mu, sigma = np.mean(values), np.std(values)
	minval, maxval = mu - sigma, mu + sigma
	width = 2. * sigma / B
	bincounts = np.zeros(B)
	for value in values:
		if value >= minval and value < maxval:
			binindex = int((value - minval) / width)
			bincounts[binindex] += 1
	return (mu, sigma, int(sum(values < minval)), bincounts) 

def median_approx(values, B):
	mu, sigma, ignorebincount, bincounts = median_bins(values, B)
	N, width = len(values), 2. * sigma / B
	index = np.argwhere(np.cumsum(bincounts) >= (N+1)/2. - ignorebincount)
	#print mu - sigma, mu + sigma, np.cumsum(bincounts), (N+1)/2. - ignorebincount, index, width
	index = index[0][0] if index.shape[0] > 0 else (B-1)
	return (mu - sigma) + (index + 1/2.) * width

def running_stats(filenames):
  '''Calculates the running mean and stdev for a list of FITS files using Welford's method.'''
  n = 0
  for filename in filenames:
    hdulist = fits.open(filename)
    data = hdulist[0].data
    if n == 0:
      mean = np.zeros_like(data)
      s = np.zeros_like(data)

    n += 1
    delta = data - mean
    mean += delta/n
    s += delta*(data - mean)
    hdulist.close()

  s /= n - 1
  np.sqrt(s, s)

  if n < 2:
    return mean, None
  else:
    return mean, s

def median_bins_fits(flist, B):
	mu, sigma = running_stats(flist)
	minval, maxval = mu - sigma, mu + sigma
	width = 2. * sigma / B
	n = 0
	for f in flist:
		hdulist = fits.open(f)
		data = hdulist[0].data
		if n == 0:
			n1, n2 = data.shape[0], data.shape[1]
			bincounts = np.zeros((n1, n2, B))
			left_bin = np.zeros_like(data)
		n += 1
		for i in range(n1):
			for j in range(n2):
				value = data[i,j]
				if value >= minval[i,j] and value < maxval[i,j]:
					binindex = int((value - minval[i,j]) / width[i,j])
					bincounts[i,j][binindex] += 1
					left_bin[i,j] += int(value < minval[i,j])
	return (mu, sigma, left_bin, bincounts) 
	
def median_approx_fits(flist, B):
	mu, sigma, left_bin, bincounts = median_bins_fits(flist, B)
	N, width = len(flist), 2. * sigma / B
	bincounts = np.cumsum(bincounts, axis=2)
	binindices = np.ones_like(bincounts)
	for index in range(B-1): 
		binindices[:,:,index] = bincounts[:,:,index] >= (N+1)/2. - left_bin
	index = np.argmax(binindices, axis=2) #np.min(np.argwhere(binindices == 1), axis=1)
	#(binindices == True)
    #print mu - sigma, mu + sigma, np.cumsum(bincounts), (N+1)/2. - left_bin, index, width
	median = (mu - sigma) + (index + 1/2.) * width
	#median = np.zeros_like(mu)
	#for i in range(median.shape[0]):
	#	for j in range(median.shape[1]):
			#index = np.argwhere(np.cumsum(bincounts[i,j,:]) >= (N+1)/2. - left_bin[i,j])
			#print mu - sigma, mu + sigma, np.cumsum(bincounts), (N+1)/2. - left_bin, index, width
			#index = index[0][0] if index.shape[0] > 0 else (B-1)
			#median[i,j] = (mu[i,j] - sigma[i,j]) + (index + 1/2.) * width[i,j]
	return median

def hms2dec(h, m, s):  
	return((1 if h >= 0 else -1)*15*(np.abs(h) + m/60. + s/(60.*60.))) #np.sign(h) 
	
def dms2dec(h, m, s):  
	return((1 if h >= 0 else -1)*(np.abs(h) + m/60. + s/(60.*60.))) #np.sign(h)

def angular_dist(ra1, dec1, ra2, dec2):
	ra1, dec1, ra2, dec2 = ra1*(np.pi/180), dec1*(np.pi/180), ra2*(np.pi/180), dec2*(np.pi/180)
	return 2*np.arcsin(np.sqrt(np.sin(np.abs(dec1 - dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(np.abs(ra1 - ra2)/2)**2))*(180/np.pi)
	#return np.degrees(2*np.arcsin(np.sqrt(np.sin(np.abs(dec1 - dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(np.abs(ra1 - ra2)/2)**2)))

def import_bss():	
	cat = np.loadtxt('bss.dat', usecols=range(1, 7))
	#print(cat[0])
	return [(i+1, hms2dec(cat[i][0],cat[i][1],cat[i][2]), dms2dec(cat[i][3],cat[i][4],cat[i][5])) for i in range(len(cat))]
	
def import_super():
	cat = np.loadtxt('super.csv', delimiter=',', skiprows=1, usecols=[0, 1])
	return [tuple([i+1] + cat[i].tolist()) for i in range(len(cat))]

def find_closest(cat, ra, dec):
	min_dist, min_id = float('Inf'), -1
	for target in cat:
		target_id, ra2, dec2 = target
		dist = angular_dist(ra, dec, ra2, dec2)
		if dist < min_dist:
			min_dist, min_id = dist, target_id
	return (min_id, min_dist)		

def crossmatch(bss_cat, super_cat, max_dist):
	matched, nonmatched = [], []
	for source in bss_cat:
		source_id, ra1, dec1 = source
		target_id, dist = find_closest(super_cat, ra1, dec1)
		if dist <= max_dist:
			matched.append((source_id, target_id, dist))
		else:
			nonmatched.append(source_id)
		#found = False
		#for target in super_cat:
		#	target_id, ra2, dec2 = target
		#	dist = angular_dist(ra1, dec1, ra2, dec2)
		#	if dist <= max_dist:
		#		matched.append((source_id, target_id, dist))
		#		found = True
		#if not found:
		#	nonmatched.append(source_id)
	return (matched, nonmatched)

def find_closest2(cat, ra, dec):
	min_dist, min_id = float('Inf'), -1
	for i in range(len(cat)):
		target_id = i
		ra2, dec2 = cat[i]
		dist = angular_dist(ra, dec, ra2, dec2)
		if dist < min_dist:
			min_dist, min_id = dist, target_id
	return (min_id, min_dist)		

def angular_dist1(ra1, dec1, ra2, dec2):
	ra1, dec1, ra2, dec2 = ra1, dec1, ra2, dec2
	return 2*np.arcsin(np.sqrt(np.sin(np.abs(dec1 - dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(np.abs(ra1 - ra2)/2)**2))

def find_closest1(cat, ra, dec):
	dists = [np.degrees(angular_dist1(ra, dec, cat[i][0], cat[i][1])) for i in range(len(cat))]
	min_id = np.argmin(dists)
	return (min_id, dists[min_id])
  
def crossmatch1(cat1, cat2, max_dist):
    
    #start = time.perf_counter()
    cat1, cat2 = np.radians(cat1), np.radians(cat2)
    matched, nonmatched = [], []
    for i in range(len(cat1)):
        source_id = i
        ra1, dec1 = cat1[i]
        target_id, dist = find_closest1(cat2, ra1, dec1)
        if dist <= max_dist:
            matched.append((source_id, target_id, dist))
        else:
            nonmatched.append(source_id)
    seconds = 0 #time.perf_counter() - start
    return (matched, nonmatched, seconds)	
	
# A function to create a random catalogue of size n
def create_cat(n):
    ras = np.random.uniform(0, 360, size=(n, 1))
    decs = np.random.uniform(-90, 90, size=(n, 1))
    return np.hstack((ras, decs))

# Write your crossmatch function here. break out
def angular_dist2(ra1, dec1, ra2, dec2):
	ra1, dec1, ra2, dec2 = ra1*(np.pi/180), dec1*(np.pi/180), ra2*(np.pi/180), dec2*(np.pi/180)
	return 2*np.arcsin(np.sqrt(np.sin(np.abs(dec1 - dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(np.abs(ra1 - ra2)/2)**2))*(180/np.pi)

def find_closest2(cat, ra, dec, max_radius):
    sort_ind = np.argsort([c[1] for c in cat])
    dec_sorted = cat[sort_ind]
    min_dist, min_id = float('Inf'), -1
    for i in range(len(dec_sorted)):
        target_id = i
        ra2, dec2 = cat[i]
        if dec2 > dec + max_radius:
          break
        dist = angular_dist2(ra, dec, ra2, dec2)
        if dist < min_dist:
            min_dist, min_id = dist, target_id
    return (min_id, min_dist)		

#     0   1   2   3   4   5   6   7   8   9
#s = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#import numpy as np
#index = np.searchsorted(s, 15, side='left')
#print(index)
	
def crossmatch2(cat1, cat2, max_dist):
    max_radius = 180
    start = time.perf_counter()
    matched, nonmatched = [], []
    for i in range(len(cat1)):
        source_id = i
        ra1, dec1 = cat1[i]
        target_id, dist = find_closest2(cat2, ra1, dec1, max_radius)
        if dist <= max_dist:
            matched.append((source_id, target_id, dist))
        else:
            nonmatched.append(source_id)
    seconds = time.perf_counter() - start
    return (matched, nonmatched, seconds)	

# Write your crossmatch function here. break out
def find_closest3(cat, ra, dec, max_radius):
    dec_sorted = np.sort([c[1] for c in cat])
    min_dist, min_id = float('Inf'), -1
    start, end = max(0, np.searchsorted(dec_sorted, dec-max_radius)), min(np.searchsorted(dec_sorted, dec+max_radius), len(dec_sorted))
    for i in range(start, end):
        target_id = i
        ra2, dec2 = cat[i]
        dist = angular_dist2(ra, dec, ra2, dec2)
        if dist < min_dist:
            min_dist, min_id = dist, target_id
    return (min_id, min_dist)		

def crossmatch3(cat1, cat2, max_dist):
    max_radius = 180
    start = time.perf_counter()
    matched, nonmatched = [], []
    for i in range(len(cat1)):
        source_id = i
        ra1, dec1 = cat1[i]
        target_id, dist = find_closest3(cat2, ra1, dec1, max_radius)
        if dist <= max_dist:
            matched.append((source_id, target_id, dist))
        else:
            nonmatched.append(source_id)
    seconds = time.perf_counter() - start
    return (matched, nonmatched, seconds)

#pip install numpy --upgrade
	
from astropy.coordinates import SkyCoord
from astropy import units as u
import re
	
def crossmatch_kd_tree(cat1, cat2, max_dist):
    
	#start = time.perf_counter()
	sky_cat1 = SkyCoord(cat1*u.degree, frame='icrs')
	sky_cat2 = SkyCoord(cat2*u.degree, frame='icrs')
	closest_ids, closest_dists, closest_dists3d = sky_cat1.match_to_catalog_sky(sky_cat2)
	matched, nonmatched = [], []
	for i in range(len(cat1)):
		source_id, target_id, dist = i, closest_ids[i], str(closest_dists[i])
		d, m, s = map(float, re.findall("\d+\.*\d*", str(dist)))
		dist = dms2dec(d, m, s)
		if dist <= max_dist:
			matched.append((source_id, target_id, dist))
		else:
			nonmatched.append(source_id)
	seconds = 0 #time.perf_counter() - start
	return (matched, nonmatched, seconds)
	
#coords1 = [[270, -30], [185, 15]]
#coords2 = [[185, 20], [280, -30]]
#sky_cat1 = SkyCoord(coords1*u.degree, frame='icrs')
#sky_cat2 = SkyCoord(coords2*u.degree, frame='icrs')
#closest_ids, closest_dists, closest_dists3d = sky_cat1.match_to_catalog_sky(sky_cat2)
#print(closest_ids)
#for d in closest_dists:
#	print(re.findall("\d+\.*\d*", str(d)))
	
def get_features_targets(data):
  # complete this function
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  return (features, data['redshift']) 
  
#data = np.load('sdss_galaxy_colors.npy')
#print(data[0])
#print(data['u'])
#print(data['u'] - data['g'])

# copy in your get_features_targets function here
def get_features_targets(data):
  # complete this function
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  return (features, data['redshift']) 

from sklearn.tree import DecisionTreeRegressor

def train_predict_regressor():
	# load the data and generate the features and targets
	data = np.load('sdss_galaxy_colors.npy')
	features, targets = get_features_targets(data)
	  
	# initialize model
	dtr = DecisionTreeRegressor()

	# train the model
	dtr.fit(features, targets)

	# make predictions using the same features
	predictions = dtr.predict(features)

	# print out the first 4 predicted redshifts
	print(predictions[:4])

def median_diff(predicted, actual):
	return np.median(abs(predicted - actual))	

# write a function that splits the data into training and testing subsets
# trains the model and returns the prediction accuracy with median_diff
def validate_model(model, features, targets):
  # split the data into training and testing features and predictions
  split = features.shape[0]//2
  train_features = features[:split]
  train_targets = targets[:split]
  test_features = features[split:]
  test_targets = targets[split:]

  # train the model
  # initialise and train the decision tree
  dtr = DecisionTreeRegressor()
  dtr.fit(train_features, train_targets)

  # get the predicted_redshifts
  # get a set of prediction from the test input features
  predictions = dtr.predict(test_features)

  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions)
  
def scatter_plot():

	data = np.load('sdss_galaxy_colors.npy')
	# Get a colour map
	cmap = plt.get_cmap('YlOrRd')

	# Define our colour indexes u-g and r-i
	u_g = data['u'] - data['g']
	r_i = data['r'] - data['i']

	# Make a redshift array
	r_s = data['redshift']

	# Create the plot with plt.scatter and plt.colorbar
	plt.scatter(u_g, r_i, c=r_s, s=5, cmap=cmap, lw=0)
	plt.colorbar()

	# Define your axis labels and plot title
	plt.xlabel('Colour idex u-g')
	plt.ylabel('Colour idex r-i')
	plt.title('Redshift (colour) u-g versus r-i')

	# Set any axis limits
	plt.xlim(-0.5, 2.5)
	plt.ylim(-0.4, 1.0)

	plt.show()

# compare the accuracy of the pediction against the actual values
#print(calculate_rmsd(predictions, test_targets))

def splitdata_train_test(data, fraction_training):
	np.random.seed(0)
	np.random.shuffle(data)
	split = int(fraction_training*data.shape[0])
	return (data[:split], data[split:])

#data = np.load('galaxy_catalogue.npy')
#for name, value in zip(data.dtype.names, data[0]):
#	  print('{:10} {:.6}'.format(name, value))

def generate_features_targets(data):
  # complete the function by calculating the concentrations

  targets = data['class']

  features = np.empty(shape=(len(data), 13))
  features[:, 0] = data['u-g']
  features[:, 1] = data['g-r']
  features[:, 2] = data['r-i']
  features[:, 3] = data['i-z']
  features[:, 4] = data['ecc']
  features[:, 5] = data['m4_u']
  features[:, 6] = data['m4_g']
  features[:, 7] = data['m4_r']
  features[:, 8] = data['m4_i']
  features[:, 9] = data['m4_z']

  # fill the remaining 3 columns with concentrations in the u, r and z filters
  # concentration in u filter
  features[:, 10] = data['petroR50_u'] / data['petroR90_u']
  # concentration in r filter
  features[:, 11] = data['petroR50_r'] / data['petroR90_r']
  # concentration in z filter
  features[:, 12] = data['petroR50_z'] / data['petroR90_z']

  return features, targets	

from sklearn.tree import DecisionTreeClassifier

# complete this function by splitting the data set and training a decision tree classifier
def dtc_predict_actual(data):
  # split the data into training and testing sets using a training fraction of 0.7
  train, test = splitdata_train_test(data, 0.7)
  # generate the feature and targets for the training and test sets
  # i.e. train_features, train_targets, test_features, test_targets
  train_features, train_targets = generate_features_targets(train)	
  test_features, test_targets = generate_features_targets(test)	

  # instantiate a decision tree classifier
  dtr = DecisionTreeClassifier()

  # train the classifier with the train_features and train_targets
  dtr.fit(train_features, train_targets)

  # get predictions for the test_features
  predictions = dtr.predict(test_features)

  # return the predictions and the test_targets
  return (predictions, test_targets)
  
# Implement the following function
def calculate_accuracy(predicted, actual):
  return sum(predicted == actual) / float(len(actual))
  
import itertools
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
 
# complete this function to get predictions from a random forest classifier
def rf_predict_actual(data, n_estimators):
  # generate the features and targets
  features, targets = generate_features_targets(data)	

  # instantiate a random forest classifier using n estimators
  rfc = RandomForestClassifier(n_estimators=n_estimators)
  
  # get predictions using 10-fold cross validation with cross_val_predict
  predictions = cross_val_predict(rfc, features, targets)

  # return the predictions and their actual classes
  return (predictions, targets)

def accuracy_by_treedepth(features, targets, depths):
  # split the data into testing and training sets
  split = features.shape[0]//2
  train_features = features[:split]
  train_targets = targets[:split]
  test_features = features[split:]
  test_targets = targets[split:]

  # initialise arrays or lists to store the accuracies for the below loop
  train_med_diffs, test_med_diffs = [], []
  
  # loop through depths
  for depth in depths:
    # initialize model with the maximum depth. 
    dtr = DecisionTreeRegressor(max_depth=depth)

    # train the model using the training set
    dtr.fit(train_features, train_targets)

    # get the predictions for the training set and calculate their median_diff
    predictions = dtr.predict(train_features)
    train_med_diffs.append(median_diff(train_targets, predictions))
    
    # get the predictions for the testing set and calculate their median_diff
    predictions = dtr.predict(test_features)
    test_med_diffs.append(median_diff(test_targets, predictions))
       
  # return the accuracies for the training and testing sets
  return (train_med_diffs, test_med_diffs)  

def cross_validate_model(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # initialise a list to collect median_diffs for each iteration of the loop below
  diffs = []
  
  for train_indices, test_indices in kf.split(features):
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    model.fit(train_features, train_targets)

    # predict using the model
    predictions = model.predict(test_features)

    # calculate the median_diff from predicted values and append to results array
    diffs.append(median_diff(test_targets, predictions))
 
  # return the list with your median difference values
  return diffs

def cross_validate_predictions(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # declare an array for predicted redshifts from each iteration
  all_predictions = np.zeros_like(targets)

  for train_indices, test_indices in kf.split(features):
    # split the data into training and testing
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
     
    # fit the model for the current set
    model.fit(train_features, train_targets)
       
    # predict using the model
    predictions = model.predict(test_features)
        
    # put the predicted values in the all_predictions array defined above
    all_predictions[test_indices] = predictions

  # return the predictions
  return all_predictions 

def split_galaxies_qsos(data):
  # split the data into galaxies and qsos arrays
  galaxies = data[data['spec_class'] == b'GALAXY']
  qsos = data[data['spec_class'] != b'GALAXY']
  # return the seperated galaxies and qsos arrays
  return (galaxies, qsos)

import psycopg2

def select_all(tbl):
  # Establish the connection
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()

  # Execute an SQL query and receive the output
  cursor.execute('SELECT * FROM ' + tbl + ';')
  records = cursor.fetchall()

  return(records)
  
#print(select_all('Star'))
#print(select_all('Planet'))  

def column_stats(tbl, col):
  # Establish the connection
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()

  # Execute an SQL query and receive the output
  cursor.execute('SELECT ' + col + ' FROM ' + tbl + ';')
  records = cursor.fetchall()
  array = np.array(records)

  return (array.mean(), np.median(array))

#print(column_stats('Star', 't_eff'))

def query(filename):
  data = np.array(np.loadtxt(filename, delimiter=',', usecols=(0,2)))
  return data[data[:,1]>1.0]
  
#result = query('stars.csv')
#print(result)  

def query2(filename):
  data = np.loadtxt(filename, delimiter=',', usecols=(0,2))
  data = data[data[:,1]>1.0]
  return data[np.argsort(data[:,1]),:]

def query3(filename1, filename2):
  data1 = np.loadtxt(filename1, delimiter=',', usecols=(0,2))
  data1 = data1[data1[:,1]>1.0]
  data2 = np.loadtxt(filename2, delimiter=',', usecols=(0,5))
  ratios = np.empty((0,1), float)
  for i in range(len(data1)):
    for j in range(len(data2)):
      if data1[i,0] == data2[j,0]:
        ratios = np.append(ratios, np.array([[data2[j,1] / data1[i,1]]]), axis=0)
  ratios.sort(axis=0)      
  return ratios
 
#t_eff = []
#for row in records:
#  t_eff.append(row[1])
#print(t_eff)
#for col in records[0]:
#    print(type(col))
#cursor.execute('SELECT radius FROM Star;')
	
# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # Run your `calc_stats` function with examples:
  print((calc_stats('data.csv'),calc_stats('data2.csv')))
  print(mean_datasets(['data4.csv', 'data5.csv', 'data6.csv']))
  print(list_stats([1.3, 2.4, 20.6, 0.95, 3.1, 2.7]))
  #print('{:.6f}s for statistics.mean'.format(time_stat(statistics.mean, 10**5, 10)))
  #print('{:.6f}s for np.mean'.format(time_stat(np.mean, 10**5, 1000)))
  print(hms2dec(23, 12, 6))
  print(dms2dec(-5, 31, 12))
  print(median_bins([1, 1, 3, 2, 2, 6], 3))
  print(median_approx([1, 1, 3, 2, 2, 6], 3))
  print(median_bins([1, 5, 7, 7, 3, 6, 1, 1], 4))
  print(median_approx([1, 5, 7, 7, 3, 6, 1, 1], 4))
  print(median_bins([0, 1], 5))
  print(median_approx([0, 1], 5))
  print(angular_dist(10.3, -3, 24.3, -29)) 
  #bss_cat = import_bss()
  #super_cat = import_super()
  #print(bss_cat)
  #print(super_cat)
  #print(find_closest(bss_cat, 175.3, -32.5))
  #matches, no_matches = crossmatch(bss_cat, super_cat, 2.75)
  #print(matches[:3])
  #print(no_matches[:3])
  #print(len(no_matches))
  #cat1 = np.array([[180, 30], [45, 10], [300, -45]])
  #cat2 = np.array([[180, 33], [55, 10], [302, -44]])
  #matches, no_matches, time = crossmatch1(cat1, cat2, 5)
  #print('matches:', matches)
  #print('unmatched:', no_matches)
  #print('time taken:', time_taken)
  cat1 = np.array([[180, 30], [45, 10], [300, -45]])
  cat2 = np.array([[180, 32], [55, 10], [302, -44]])
  matches, no_matches, time_taken = crossmatch1(cat1, cat2, 2.75)
  print 'naive cross-match'
  print(matches[:3])
  print(no_matches[:3])
  print(len(no_matches))
  print 'kd-tree cross-match'
  matches, no_matches, time_taken = crossmatch_kd_tree(cat1, cat2, 5)
  print(matches[:3])
  print(no_matches[:3])
  print(len(no_matches))
  
  import pandas as pd
  df = pd.read_csv('C:\\courses\\coursera\\Past\\ML\\Data Driven Astronomy\\sdss_galaxy_catalog.csv')
  #df.head()

  from sklearn.metrics import confusion_matrix
  from sklearn.model_selection import cross_val_predict
  from sklearn.tree import DecisionTreeClassifier, export_graphviz
  from sklearn import grid_search
  
  parameters = {'max_depth':range(3,20)}
  dtc = grid_search.GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
  dtc.fit(X=df.iloc[:, :-1], y=df['class'])
  print (dtc.best_score_, dtc.best_params_) 
  dtc = dtc.best_estimator_
  
  #features, targets = generate_features_targets(df)
  #dtc = DecisionTreeClassifier()
  #dtc.fit(train_features, train_targets)
  # get the predictions for the training set and calculate their median_diff
  #predictions = dtr.predict(train_features)
    
  #predicted = cross_val_predict(dtc, features, targets, cv=10)
  from StringIO import StringIO
  dotfile = StringIO()
  export_graphviz(dtc, out_file=dotfile)#,feature_names=['u - g', 'g - r', 'r - i', 'i - z'])
  import pydotplus as pydotplus
  graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
  graph.write_png("C:\\courses\\coursera\\Past\\ML\\Data Driven Astronomy\\decision_tree.png")
    
  '''
  # load the data
  data = np.load('sdss_galaxy_colors.npy')
  print(data.dtype.names)
  f=open('sdss_galaxy_colors.csv','wb')
  np.savetxt(f, data, fmt='%s', newline='\r\n') 
  np.savetxt(f, data[:30000], fmt='%s', newline='\r\n',delimiter=',') 
  #np.savetxt(f, features, fmt='%s', newline='\r\n') 
  np.savetxt(f, targets, fmt='%s', newline='\r\n') 
  np.np.savez_compressed(f, data, fmt='%s', newline='\r\n') 
  f.close()
  import glob, os
  for f in glob.glob("*.csv"): os.remove(f)
  for f in glob.glob("*.npy"): os.remove(f)
  # split the data
  features, targets = generate_features_targets(data)

  # train the model to get predicted and actual classes
  dtc = DecisionTreeClassifier()
  predicted = cross_val_predict(dtc, features, targets, cv=10)

  # calculate the model score using your function
  model_score = calculate_accuracy(predicted, targets)
  print("Our accuracy score:", model_score)

  # calculate the models confusion matrix using sklearns confusion_matrix function
  class_labels = list(set(targets))
  model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

  # Plot the confusion matrix using the provided functions.
  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()
  '''