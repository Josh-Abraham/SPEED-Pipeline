#########################################################################
#                                                                       #
#                              SPEED Pipeline                           #
#                           By: Joshua Abraham                          #
#                                                                       #
#########################################################################

import argparse
import cv2
import os
import glob
import imageProcessing
import HEDmodel
from warnings import simplefilter
# ignore all future warnings because of medpy
simplefilter(action='ignore', category=FutureWarning)

def loadImage(location):
	# Import image from loaction, as well as the shape of matrix
	image = cv2.imread(location)
	(H, W) = image.shape[:2]

	scale_percent = 100
	width = int(W * scale_percent / 100)
	height = int(H * scale_percent / 100)
	dim = (width, height)

	# resize image
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)	

	return [image, height, width]

def plotSimilarities(similarity):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	SMALL_SIZE = 12
	SMALL_MED = 13
	MEDIUM_SIZE = 14
	BIGGER_SIZE = 16

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=SMALL_MED)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	x = []
	y = []
	z = []
	for i in range(len(similarity[0])):
		x.append(i)
		y.append(i+0.5)
		z.append(i+1)
	
	colormap = ['seagreen', 'darkslateblue', 'silver']
	labelmap = ['Unprocessed', 'HED', 'Preprocessed']
	alpha = [1, 1, 1]
	points = [x, y, z]
	width = 0.15
	i = 0
	for similaritySet in similarity:
		ax.bar(points[i], similaritySet, width, align='center', color=colormap[i], alpha=alpha[i], label=labelmap[i])
		i = i + 1
	# Plot
	my_xticks = []
	plt.xticks(x, my_xticks)
	ax.legend(loc=0)
	plt.title('Edge Detection Algorithm Comparison')
	plt.xlabel('Edge Detection')
	plt.ylabel('Similarity (%)')
	plt.show()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge-detector", type=str, required=True, help="path to OpenCV's deep learning edge detector")
ap.add_argument("-i", "--path", type=str, required=True, help="path to input file")
args = vars(ap.parse_args())

# Create Network Connections based off trained model
protoPath = os.path.sep.join([args["edge_detector"],
"deploy.prototxt"])
modelPath = os.path.sep.join([args["edge_detector"],
	"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# register our new layer with the model holding our desired image size
cv2.dnn_registerLayer("Crop", HEDmodel.HEDmodel)


path = args["path"] + '/*.png'
listOfGT = ['Images/GroundTruth']
listOfPaths = ['Images/Original']

for pathPart in listOfPaths:
	path = pathPart + '/*.png'
	verticalStitch = []
	# Unprocessed, HED, Preprocessed
	similaritySets = [[], [], []]
	imageObjects = []
	i = 0
	j = 0
	x = 0
	missed_canny = []
	missed_HED = []
	missed_SPEED = []
	extra_canny = []
	extra_HED = []
	extra_SPEED = []
	for filename in glob.glob(path): #assuming png
		groundTruth_filename = listOfGT[0] + '\\' + filename.split('\\')[1].split('.')[0] + '_gt.png'
		
		[image, H, W] = loadImage(filename)
		[gt, H, W] = loadImage(groundTruth_filename)
		imageObject = imageProcessing.image(image, gt, H, W, net, i)
		simCalc = imageObject.runPreprocessing()
		imageObject.showImages()