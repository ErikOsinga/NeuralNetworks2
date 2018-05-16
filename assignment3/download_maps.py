import numpy as np
from urllib import request
import sys

# For running on local pc
data_dir = '/home/erik/Desktop/vakken/NeuralNetworks/maps_data/'

# for running on Duranium
# data_dir = '/data/s1546449/maps_data_28_zoom15/'
# data_dir = '/data/s1546449/maps_data_28_zoom12_2/'
# data_dir = '/data/s1546449/maps_data_200_zoom15/'
# data_dir = '/data/s1546449/maps_data_validation200/'
data_dir = '/data/s1546449/maps_data_validation512/'


def generate_link(size,lat,lon,zoom,maptype='roadmap'):
	'''
	Function to generate a link to a certain google maps cutout.

	size --> integer which determines the amount of pixels in a squared image
	lat --> latitude of the center (float) #(Up and Down)
	lon --> longitude of the center (float) #(Left and Right)
	zoom --> zoom factor, integer
	maptype --> 'roadmap' or 'satellite'

	'''
	begin = "http://maps.google.com/maps/api/staticmap?sensor=false"
	end = "&style=feature:all|element:labels|visibility:off"
	# key = "&key=AIzaSyD7o6a-MKAtFX08sqHG-_Vkk8OShV6oJmY"

	# Petros key
	key = "&key=AIzaSyBcVLuX2eeRNJ2k0yrW99XBDQ7dENPTs4M" 
	
	size = str(size)+'x'+str(size)
	center = str(lat)+','+str(lon)
	zoom = str(zoom)

	if maptype not in ['roadmap' , 'satellite']:
		raise ValueError("Wrong maptype")

	link = begin+'&size=%s&center=%s&zoom=%s'%(size,center,zoom)+end+'&maptype=%s'%maptype+key

	return link

def download_from_link(link,filename):
	"""
	Saves PNG images from a link
	"""

	request.urlretrieve(link,data_dir+filename+'.png')

def download_images(number_of_images,num_already_downloaded,size,zoom):
	'''
	Download a square root-able number of images 
	'''

	if int(number_of_images**0.5)**2 != number_of_images:
		raise ValueError("Please give number of images that can be sq. rooted")

	# Number of images per direction 
	num_im = int(number_of_images**0.5) 
	
	'''
	# go from 52.1688731, 4.4569086 approx Leiden
	starting_lat = 47.792078
	ending_lat = 52.1688731
	
	# to      47.792078, 13.189666 approx Salzburg 
	starting_lon = 4.4569086
	ending_lon = 13.189666
	'''

	# New square
	# go from 52, 13 approx berlin, to Roemenie
	# starting_lat = 47.968274
	# ending_lat = 52.893429 

	# starting_lon = 13.768414
	# ending_lon = 22.931304 


	# New squae
	# go from approx Birmingham to Eastbourne
	# starting_lat = 50.856039
	# ending_lat = 52.763742 
	# starting_lon = -2.293847
	# ending_lon = 0.279611


	# New square: 
	# Leiden
	# starting_lat = 52.149310
	# ending_lat = 52.170210
	# starting_lon = 4.455526
	# ending_lon =  4.508515

	# New square:
	# Amsterdam
	starting_lat = 52.346097
	ending_lat = 52.384455
	starting_lon = 4.872128
	ending_lon =  4.954774

	'''Note: '''
	# For zoom 15 and size 512x512 the increments must be 
	# increment_lat = 0.003200
	# increment_lon = 0.005450

	# Because first num_already_downloaded images have already been downloaded:
	# redefine starting and ending coordinates

	print ('Downloading %i images, starting at image number %i'%(number_of_images
                                                ,num_already_downloaded+1) )
	i = num_already_downloaded + 1
	for lat in np.linspace(starting_lat,ending_lat,num_im):
		for lon in np.linspace(starting_lon,ending_lon,num_im):
			image_number = '%06i' % i
			
			link = generate_link(size=size,lat=lat,lon=lon,zoom=zoom,maptype='satellite')
			
			filename = image_number+'_satellite'
			download_from_link(link,filename)

			link = generate_link(size=size,lat=lat,lon=lon,zoom=zoom,maptype='roadmap')
			
			filename = image_number + '_roadmap'
			download_from_link(link,filename)

			sys.stdout.write("\r%i"%i)
			sys.stdout.flush()

			i += 1


# Note: # Each image is about 500m (for the zoom 15, 512x512x3 pixel ones)

number_of_images = 100 # actually downloading twice as much, since we 
						# have a satellite and roadmap for every image

print ('Saving to data directory: ', data_dir)
download_images(number_of_images,num_already_downloaded=100,size=512,zoom=15)
