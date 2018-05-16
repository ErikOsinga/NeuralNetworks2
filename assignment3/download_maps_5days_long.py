import numpy as np
from urllib import request
import sys
import time

# For running on local pc
data_dir = '/home/erik/Desktop/vakken/NeuralNetworks/maps_data/'

# for running on Duranium
data_dir = '/data/s1546449/maps_data_200_zoom15/'

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
	key = "&key=AIzaSyD7o6a-MKAtFX08sqHG-_Vkk8OShV6oJmY"
	
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

	# go from 52.1688731, 4.4569086 approx Leiden
	starting_lat = 47.792078
	ending_lat = 52.1688731
	
	# to      47.792078, 13.189666 approx Salzburg 
	starting_lon = 4.4569086
	ending_lon = 13.189666

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

			if i == 12100:
				print ('Daily limit reached, sleeping for a day.. zzz')
				time.sleep(24*60*60)


# Note: # Each image is about 500m (for the zoom 15, 512x512x3 pixel ones)

number_of_images = 120409 # actually downloading twice as much, since we 
						# have a satellite and roadmap for every image

print ('Saving to data directory: ', data_dir)
download_images(number_of_images,num_already_downloaded=0,size=200,zoom=15)
