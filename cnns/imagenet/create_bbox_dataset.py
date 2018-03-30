# John Lambert

import os
import glob
import shutil

# for mask plotting
import numpy as np
import cv2

from cnns.data_utils.localization_utils import _load_pascal_annotation

PATH_TO_IMAGENET_2012_DIRECTORY = '/vision/group/ImageNet_2012/'
PATH_TO_IMAGENET_LOCALIZATION_DIRECTORY = '/vision/group/ImageNetLocalization/'

# Code to find all images with corresponding bounding box annotations.
# Create a new dataset with only these images.

def build_synset_to_classname_dict():
    fname = 'synset_to_classname.txt'
    with open(fname) as f:
        content = f.readlines()
    dict = {}
    for line in content:
        mapping_tuple = line.split('\t')
        dict[ mapping_tuple[0] ] = mapping_tuple[1]
    return dict


class LocalizationDataLoader(object):
    def __init__(self):

        self.synset_to_classname_dict = build_synset_to_classname_dict()

        self.use_bw_mask = False # later change this to True
        # OLD LOCATIONS
        self.imagenet_path = PATH_TO_IMAGENET_2012_DIRECTORY
        self.train_set_path = os.path.join( self.imagenet_path, 'train')
        self.val_set_path = os.path.join( self.imagenet_path, 'val')

        self.train_annotation_path = os.path.join(self.imagenet_path, 'TrainAnnotation' ) #''Annotation/')
        self.val_annotation_path = os.path.join( self.imagenet_path, 'ValAnnotation/val' )

        # NEW LOCATIONS
        self.localization_path = PATH_TO_IMAGENET_LOCALIZATION_DIRECTORY
        self.new_localization_train_path = os.path.join( self.localization_path, 'train' )
        self.new_localization_val_path = os.path.join( self.localization_path, 'val' )
        self.new_localization_annotation_path = os.path.join( self.localization_path, 'localization_annotation' )

        self.train_annotation_synsets = os.listdir( self.train_annotation_path )
        self.val_annotation_synsets = os.listdir( self.val_annotation_path )

        self.compare_split_annotation_synsets()

        self.train_synsets = os.listdir(self.train_set_path)
        self.val_synsets = os.listdir(self.val_set_path)

        self.compare_train_val_synsets()

        self.train_xml_dict = {}
        self.val_xml_dict = {}
        self.get_split_xml_dict('train')
        self.get_split_xml_dict('val')

        self.train_im_dict = {}
        self.val_im_dict = {}
        self.get_split_im_dict('train')
        self.get_split_im_dict('val')

        self.matches = []
        self.write_split_matches( 'train', self.train_im_dict, self.train_set_path)
        self.write_split_matches( 'val', self.val_im_dict, self.val_set_path)

        print 'Data set preparation complete.'

    def compare_split_annotation_synsets(self):
        print 'Train annotation synsets has len: ', len( self.train_annotation_synsets )
        print 'Val annotation synsets has len: ', len( self.val_annotation_synsets )
        for train_synset in self.train_annotation_synsets:
            if train_synset not in self.val_annotation_synsets:
                print 'We have a massive error: ', train_synset, ' not in val_annotation_synsets, but in train_annotation_synsets.'

    def get_split_xml_dict(self, split):
        if split == 'train':
            xml_dict = self.train_xml_dict
            split_annotation_path = self.train_annotation_path
        elif split == 'val':
            xml_dict = self.val_xml_dict
            split_annotation_path = self.val_annotation_path
        for i, synset in enumerate(self.train_annotation_synsets):
            print i
            synset_xml_files = glob.glob1(os.path.join(split_annotation_path, synset), '*.xml')
            xml_dict[synset] = synset_xml_files
        print split, 'annotation run-through complete.'
        print len(xml_dict.keys() )


    def compare_train_val_synsets(self):
        print 'Train synsets has len: ', len(self.train_synsets)
        print 'Val synsets has len: ', len(self.val_synsets)
        for train_synset in self.train_synsets:
            if train_synset not in self.val_synsets:
                print 'We have a massive error: ', train_synset, ' not in val_synsets.'


    def get_split_im_dict(self, split):
        if split == 'train':
            split_path = self.train_set_path
            split_dict = self.train_im_dict
        elif split == 'val':
            split_path = self.val_set_path
            split_dict = self.val_im_dict
        for i, synset in enumerate(self.train_synsets): # val and train have same synsets!
            print i
            synset_im_files = glob.glob1(os.path.join(split_path, synset), '*.JPEG')
            split_dict[synset] = synset_im_files
        print split, ' set run-through complete.'
        print len(split_dict.keys() )


    def write_split_matches(self, split, split_dict, old_split_path):
        """ """
        if split == 'train':
            split_xml_dict = self.train_xml_dict
            annotation_path = self.train_annotation_path
        elif split == 'val':
            split_xml_dict = self.val_xml_dict
            annotation_path = self.val_annotation_path

        for synset,im_filename_list in split_dict.iteritems():
            new_synset_im_dir = os.path.join( self.localization_path, split, synset )
            #if not os.path.isdir( new_synset_im_dir ):
                # os.makedirs( new_synset_im_dir ) # mkdir for this synset in new localization location
            #    print 'Making dir: ', new_synset_im_dir
            new_synset_xml_dir = os.path.join( self.new_localization_annotation_path, synset )
            #if not os.path.isdir( new_synset_xml_dir ):
                # os.makedirs( new_synset_xml_dir ) # mkdir for this synset in new localization location
            #    print 'Making dir: ', new_synset_xml_dir
            print synset
            for i, file_name in enumerate(im_filename_list):

                #if split == 'train':
                file_name_stem = file_name.split('.')[0]
                # elif split == 'val':
                #     file_name_stem = synset + '_' + file_name.split('_')[2]
                for annotation_file_name in split_xml_dict[synset]:
                    annotation_file_name_stem = annotation_file_name.split('.')[0]
                    if file_name_stem == annotation_file_name_stem:
                        #print 'Match: ', file_name_stem, annotation_file_name_stem
                        #self.matches.append( annotation_file_name )
                        old_im_path = os.path.join( old_split_path, synset, file_name )
                        im = cv2.imread( old_im_path)
                        if im.shape[2] != 3:
                            print 'NOT 3 CHANNEL IMAGE! MASSIVE ERROR'
                            print im.shape
                        #if (im.shape[0] < 224) or (im.shape[1] < 224):
                        #    print 'less than 224 px in one dim IMAGE! MASSIVE ERROR'
                        #    print im.shape

                        new_im_path = os.path.join( new_synset_im_dir, file_name )
                        #print 'Would be copying im from ', old_im_path, ' to: ', new_im_path
                        # shutil.copy( old_im_path, new_im_path) # for the xml file

                        old_xml_path = os.path.join( annotation_path, synset, annotation_file_name )
                        new_xml_path = os.path.join( self.new_localization_annotation_path, synset, \
                                                     annotation_file_name )
                        #print 'Would be copying xml from ', old_xml_path, ' to: ', new_xml_path
                        #rand_num = random.uniform(1, 1000)
                        #if rand_num < 5: # .5% chance of occurring
                        #    self.write_original_im_vs_dets_plots( old_im_path, old_xml_path, synset, i )
                        # shutil.copy( old_xml_path, new_xml_path ) # for the image file
        #print 'We found ', len(self.matches), ' matches for the ', split, ' split.'


    def write_original_im_vs_dets_plots(self, im_file_path, xml_file_path, synset, idx):
        """ sample random images"""
        dets = _load_pascal_annotation(xml_file_path)
        self.save_side_by_side( self.use_bw_mask, im_file_path, dets, synset, idx)

    def save_side_by_side(self, use_bw_mask, fname, dets, synset, idx):
        """
        dets =  [ 	(100,100,200,350),
                    (350,200,100,250)     ]
        """
        # read in ImageNet image
        im = cv2.imread(fname)
        im_h, im_w, im_c = im.shape
        # read in its x,y,w,h coordinate detections
        mask_im = np.zeros((im_h, im_w, im_c)).astype(np.uint8)
        for det in dets:
            x1, y1, x2, y2 = det  # read in the detections
            if use_bw_mask:
                mask_im[y1:y2, x1:x2, :] = 255
            else:
                mask_im[y1:y2, x1:x2, :] = im[y1:y2, x1:x2, :]
        # return the image, with everything outside of the union as black
        vis = np.concatenate((im, mask_im), axis=1)
        classname = self.synset_to_classname_dict[synset]
        sample_im_savename = 'sample_' + classname + '_' + synset + '_' + str(idx) + '.png'
        cv2.imwrite(sample_im_savename, vis)


if __name__ == '__main__':
    data_loader = LocalizationDataLoader()



    # for key, val in self.xml_dict.iteritems():
    #     if key in self.train_im_dict:
    #         print 'Match: synset = ', key
    #         self.matches.append( key )
    # print 'We found ', len(self.matches), ' matches.'