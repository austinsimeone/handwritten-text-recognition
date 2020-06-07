from __future__ import division
from __future__ import print_function



import numpy as np
from data import preproc as pp
import pandas as pd
import os

class Sample:
    "single sample from the dataset"
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath
        
class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class DataLoader:
    "loads data which corresponds to IAM format"

    def __init__(self, filePath, batchSize, imgSize, maxTextLen,train = True):
        "loader for dataset at given location, preprocess images and text according to parameters"
        
        #make the end of the filepathlist contain the / so that we can add the file name to the end of it
        
        #will me augment the data in anyway?
        self.dataAugmentation = False
        #where does the index start - should always be 0
        self.currIdx = 0
        #self selected batch size
        self.batchSize = batchSize
        #X & Y coordinates of the png
        self.imgSize = imgSize
        #empty list of images to fill with the samples
        self.samples = []
        self.filePath = filePath
        self.maxTextLen = maxTextLen
        self.partitionNames = ['trainSample','validationSample']
        self.train = train
        
        df = pd.read_csv('/home/austin/Documents/Github/SimpleHTR/words_csv/2020-06-03 11:39:42.000901.csv')
        chars = set()
        for index, row in df.iterrows():
            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileName = row['file_name']
            # GT text are columns starting at 9
            gtText = row['truth']
            chars = chars.union(set(list(gtText)))
            # put sample into list
            self.samples.append(Sample(gtText, fileName))

        # split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        trainSamples = self.samples[:splitIdx]
        validationSamples = self.samples[splitIdx:]

        # put words into lists
        trainWords = [x.gtText for x in trainSamples]
        validationWords = [x.gtText for x in validationSamples]
        
        self.img_partitions = [trainSamples,validationSamples]
        self.word_partitions = [trainWords,validationWords]


        # number of randomly chosen samples per epoch for training 
        self.numTrainSamplesPerEpoch = 25000 


        # list of all chars in dataset
        self.charList = sorted(list(chars))
        
        self.train_steps = int(np.ceil(len(self.word_partitions[0]) / self.batchSize))
        self.valid_steps = int(np.ceil(len(self.word_partitions[1]) / self.batchSize))
       
    def truncateLabel(self, text):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input 
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > self.maxTextLen:
                return text[:i]
        return text


    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)


    def getNext(self):
        "iterator"
        if self.train == True:
            j = 0
        else:
            j = 1
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.img_partitions[j][i].gtText for i in batchRange]
        imgs = [pp.preprocess(os.path.join(self.filePath,self.img_partitions[j][i].filePath),self.imgSize) for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)


