"""Dataset reader and process"""

import os

import multiprocessing
from tqdm import tqdm
from data import preproc as pp
from functools import partial


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        #this is the path to the data
        self.source = source
        #this is the dataset type. can probably be removed later
        self.name = name
        #dataset created from the two file types
        self.dataset = None
        #three different partitions for model development
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = dict()

            for y in self.partitions:
                self.dataset[y] = {'dt': [], 'gt': []}

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def preprocess_partitions(self, input_size):
        """Preprocess images and sentences from partitions"""

        for y in self.partitions:
            arange = range(len(self.dataset[y]['gt']))

            for i in reversed(arange):
                text = pp.text_standardize(self.dataset[y]['gt'][i])

                if not self.check_text(text):
                    self.dataset[y]['gt'].pop(i)
                    self.dataset[y]['dt'].pop(i)
                    continue

                self.dataset[y]['gt'][i] = text.encode()

            results = []
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                print(f"Partition: {y}")
                for result in tqdm(pool.imap(partial(pp.preprocess, input_size=input_size), self.dataset[y]['dt']),
                                   total=len(self.dataset[y]['dt'])):
                    results.append(result)
                pool.close()
                pool.join()

            self.dataset[y]['dt'] = results

    def _custom(self):
        """ custom data that i have generated """
        source = os.path.join(self.source,'custom_source')
        
        paths = {"images": os.path.join(source,'words_screenshot_labeled'),
                 "text": os.path.join(source, 'words_csv')}
        
        
        # need to create a dictionary of the file path and the ground truth
        # dataset MUST BE A DICTIONARY
        # want to add the randomization of test train split
        # seeding with python? 


    @staticmethod
    def check_text(text):
        """Make sure text has more characters instead of punctuation marks"""

        strip_punc = text.strip(string.punctuation).strip()
        no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

        if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
            return False

        punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

        return len(no_punc) > 2 and punc_percent <= 0.1
