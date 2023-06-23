# imports

import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import time

import copy

from astropy.io import fits
from astropy.wcs import WCS

from scipy import ndimage, misc



# ------------------------------- Image Class ---------------------------------#
class image:
    def __init__(self, filename):
        self.filename = filename
        
        self.file = fits.open(self.filename)
        self.f = copy.deepcopy(self.file)
        self.file.close()
        
        self.info = {}
        self._extract_info()
            
            
    def _extract_info(self):
        """
        Save the information related to type, date, time and distance of the picture
        """
        
        if 'geo' in self.filename:
            self.info['type'] = 'Geometry'
            
        elif 'l2b' in self.filename:
            self.info['type'] = 'Picture'
        
        underscores_index = []
        for i in range(len(self.filename)):
            if self.filename[i] == '_':
                underscores_index.append(i)
        
        # the information is in the file name and separated by underscores, so we use them so isolate the info of interest
        self.info['date'] = self.filename[underscores_index[1] + 1 : underscores_index[2]]
        self.info['time'] = self.filename[underscores_index[2] + 1 : underscores_index[3]]
        self.info['distance'] = self.f[1].header['S_DISTAV']
           
                
    def to_array(self, i=1):
        """
        Convert the UV or Geometry information into a multidim array
        """
                
        if self.info['type'] == 'Picture':
                
            img = self.f[1].data
            
            # Remove eventual negative numbers 
            index = np.where(img < 0)
            img[index]=0
            
        elif self.info['type'] == 'Geometry':
            img = self.f[i].data
            
        return img
    
    def show_header(self):
        """
        Show the name of the features
        """
        
        print(self.f[1].header)
    
    def show_data(self, i=1):       
        """
        Show the selected data, UV for the pictures or one of the 8 gemetrical information
        """
        
        if self.info['type'] == 'Picture':
            img = self.f[1].data
            
            # Remove eventual negative numbers 
            index = np.where(img < 0)
            img[index]=0
            
            plt.imshow(img, cmap='gray')
            
            #plt.savefig('your-file.png', dpi=400, quality=95)
            
        elif self.info['type'] == 'Geometry':
            infos = ['latitude', 
                     'longitude', 
                     'local time', 
                     'phase angle', 
                     'incidence angle', 
                     'emission angle', 
                     'azimuthal angle']
            print(infos[i-1])
            img = self.f[i].data
            
            plt.imshow(img, cmap='gray')
            
            
            
# ------------------------------- Data Processing ---------------------------------#
def load_data(rootdir):
    """
    Load all the files contained in the nested subfolders of the dataset
    """
    
    imgs = []
    for subdir, dirs, files in sorted(os.walk(rootdir, topdown=True)):
        
        print('subdir: ', subdir)
        
        imgs_same_folder = []
        if len(files)>0:
            for file in sorted(files):
                filepath = os.path.join(subdir, file)
                if filepath.endswith('.fit'):
                    img = image(filepath)
                    imgs_same_folder.append(img)
            imgs.append(imgs_same_folder)
            
    return imgs


def match_uv_to_geo(geo_files, uv_files):
    """
    Create couples of UV and Geometry files 
    """
    
    # make a list of all the geometry files
    complete_geo_files = []
    for folder in geo_files:
        for x in folder:
            complete_geo_files.append(x)
       
    # make a list of all the UV files
    complete_uv_files = []
    for folder in uv_files:
        for x in folder:
            complete_uv_files.append(x)
    
    matched_files = []
 
    for x in complete_geo_files:
        for y in complete_uv_files:
            # print(" date: ", x.info['date'], " time: ", x.info['time'],  "date: ", y.info['date'], " time: ", y.info['time'])
            
            # if the date and time match
            if x.info['date'] == y.info['date'] and x.info['time'] == y.info['time']:
                # create a couple
                match = [x, y]
                # add the couple to the list of matching couples
                matched_files.append(match)
                # remove the current uv file
                complete_uv_files.remove(y)
                
                break                 
                
    return matched_files
                                     
                
# ------------------------------- Data Visualisation ---------------------------------#
def multiple_plot(imgs, n_columns = 10, verbose = False):
    """
    Create a multiple plots for the UV pictures
    """
    
    for num, img in enumerate(imgs):

        if verbose == True:
            print(img.info['date'], ' ', img.info['time'])
        
        rows = int(len(imgs)/n_columns) + 1
        
        plt.subplot(rows, n_columns, num+1)
        plt.axis('off')
        plt.imshow(img.to_array(), cmap='gray')
        

        
# ------------------------------- Find Square ---------------------------------#        
def find_square(matrix, m, degrees):
    """
    Inefficient algorithm to crop the biggest possible square from a UV picture
    """
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    skip = 1
    area_max = (0, [])

    a = matrix
    w = np.zeros(dtype=int, shape=a.shape)
    h = np.zeros(dtype=int, shape=a.shape)

    # for every row
    for r in range(nrows):
    
        # for every column
        for c in range(ncols):
        
            # if it's a one, skip
            if a[r][c] == skip:
                continue
            
            # if it's the first row
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r-1][c]+1
            
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c-1]+1
            
            minw = w[r][c]
                
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                                
                # if condition is for a square
                if (minw == dh+1) and (minw == degrees):
                
                    area = (dh+1)*minw
                    if area > area_max[0]:
                        
                        area_max = (area, [(r-dh, c-minw+1, r, c)])
                        
    if area_max[0]> 5:

        return area_max[1][0]
    else:
        return None
    
    
# ------------------------------- Remove Square ---------------------------------#        
def remove_square(a, r0, c0, r1, c1):
    """
    Remove square from the UV picture, since now it's been cropped
    """
    for r in range(r0, r1):
        for c in range(c0, c1):
            a[r][c] = 1
    return a
                




