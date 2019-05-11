import glob
import numpy as np 
from PIL import Image
from itertools import combinations
import os
import argparse

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument("path", help="path to the folder with images",
	                    type=str)

	parser.add_argument("--hash_threhsold", default=6,
	                    help='''
	                    Threhsold for determing similar pictures using median image hashing algorithm.
	                    Hash image is resized to have 8x8 dimension (i.e. 64 values)''',
	                    type=int)

	parser.add_argument("--histogram_similarity", default=.9,
	                     help='''
	                    Threhsold for determing similar pictures using color histogram.
	                          ''',
	                    type=float)

	args = parser.parse_args()

	return args

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def median_image_hashing(image_path, hash_dims=(8,8)):

    '''
    Median image hashing algorithm

    By courtesy of:
    http://hackerfactor.com/blog/index.php%3F/archives/432-Looks-Like-It.html
    '''
    image_pil = Image.open(image_path)
    resized_im = np.array(image_pil.resize(hash_dims))
    gray_im = rgb2gray(resized_im)
    binarized = (gray_im > np.median(gray_im)).astype(np.uint8)
    average_hash = binarized.flatten()
    return average_hash

def hamming_distance(hash_pair):
    return sum((hash_pair[0]!=hash_pair[1]).astype(np.uint8))



def color_histogram(image_path, n_bins=20, dims = (112,112)):
    '''
    Image is read, resized(according to dims arg) and decolorized first.
    Histogram of pixel values is created then.
    '''
    image_pil = Image.open(image_path)
    resized_im = np.array(image_pil.resize(dims))
    gray_im = rgb2gray(resized_im)
    return np.histogram(gray_im, bins=np.linspace(0, 255, n_bins))[0] 



def histogram_intersection(hist_pair):

    minima = np.minimum(hist_pair[0], hist_pair[1])
    intersection = np.true_divide(np.sum(minima), np.sum(hist_pair[1]))
    return intersection 


def find_dups_from_dir(data_dir, hash_threshold=6, hist_threshold=.9):
  
    if not os.path.exists(data_dir):
        print('Bad boy')
        return  

    images_filenames = np.array(glob.glob(data_dir+'/*'))
    possible_ind_combs = np.array(list(combinations(range(len(images_filenames)),2)))
    image_hashes = np.array([median_image_hashing(path) for path in images_filenames])
    color_distribution = np.array([color_histogram(path) for path in images_filenames])
    hash_pairs, dist_pairs = image_hashes[possible_ind_combs], color_distribution[possible_ind_combs]
    hamming_distances = np.array(list(map(hamming_distance, hash_pairs)))
    histogram_intersections =  np.array(list(map(histogram_intersection, dist_pairs)))
    
   
    mask1 = hamming_distances<hash_threshold
    mask2 = histogram_intersections>hist_threshold
    final_mask = mask1 | mask2
    dup_ind = np.argwhere(final_mask)
    image_ind = possible_ind_combs[dup_ind]
    
    return np.squeeze(images_filenames[image_ind])


def main():
	args = parse_args()
	dups = find_dups_from_dir(args.path, args.hash_threhsold, args.histogram_similarity)
	print(dups)

if __name__ == '__main__':
    main()