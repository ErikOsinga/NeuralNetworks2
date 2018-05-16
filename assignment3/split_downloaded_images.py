import sys
import numpy as np

import glob
import matplotlib.image as mpimg
import scipy.misc

def split_image(sat_image,road_image,num_images,new_image_dir,counter):
	"""
	Splits one image into smaller ones

	sat_image -- numpy array, the satellite image to split
	road_image -- numpy array, the road image to split
	num_images -- integer, the number of images the image is split into
	new_image_dir -- where to save the new image
	counter -- for naming.
	"""

	# number of images the original image is split into
	image_size = sat_image.shape[0]
	new_im_size = int(image_size/num_images)
	sys.stdout.write("\rImage number %i"%counter)
	sys.stdout.flush()
	for i in range(num_images-1): # -1 to get stop before the Google Logo at the bottom
		for j in range(num_images):
			new_sat_image = sat_image[new_im_size*i:new_im_size*(i+1), new_im_size*j:new_im_size*(j+1)]
			new_road_image = road_image[new_im_size*i:new_im_size*(i+1), new_im_size*j:new_im_size*(j+1)]
			scipy.misc.imsave('%s/%i_%i_%i_satellite.png'%(new_image_dir,counter,i,j) ,new_sat_image)
			scipy.misc.imsave('%s/%i_%i_%i_roadmap.png'%(new_image_dir,counter,i,j) ,new_road_image)

def split_images(image_size,new_image_size,image_dir,new_image_dir,amount):
	"""
	Splits images into smaller ones

	image_size -- original image size
	new_image_size -- size of the new smaller images
	image_dir -- directory of the large images
	new_image_dir -- directory of the smaller images
	amount -- the amount of images in the directory to split into smaller images
	"""

	print (' ================================ ')
	num_images = int(image_size/new_image_size)
	print ('Will split %i original images with size %i into %i images with new image size %i'
		%(amount,image_size,num_images**2,new_image_size))
	print ('Original images from %s, new images into %s'%(image_dir,new_image_dir))
	print (' ================================ ')

	counter = 0
	for imname in sorted(glob.glob(image_dir+'*_satellite.png')):
		
		sat_image = mpimg.imread(imname)
		road_image = mpimg.imread(imname.replace('satellite','roadmap'))

		split_image(sat_image,road_image,num_images,new_image_dir,counter)

		counter += 1
		if counter == amount:
			break


image_dir = '/data/s1546449/maps_data/'
new_image_dir = '/data/s1546449/maps_data_28_zoom15_split/'
amount = 17800//2

split_images(512,28,image_dir,new_image_dir,amount)