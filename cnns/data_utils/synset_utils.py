import cPickle as pkl

def get_idx2synset_dict():
    """ read python dict back from the file """
    f = open('imagenet_idx2class.pkl', 'rb')
    dict = pkl.load(f)
    f.close()
    return dict
