#Understanding the problem statement - 
#Build a deep learning algorithm to automatically classify different music genres 
#from audio files. This is done by using low-level frequency and time domain. 

#loading the dataset - 
#for this project, we need a dataset of audio tracks having similar size and similar 
#frequency range. We will use GTZAN genre classification dataset. 
#GTZAN genre colletion dataset was collected in 2000-01. It consists of 1000 audio files each having 
#30 seconds duration. There are 10 music genres each having 100 audio tracks. each track having .wav 
#format. It contains audio files from the 10 following genres - 
#1. Blues 
#2. Classical 
#3. Country 
#4. Disco 
#5. Hiphop 
#6. Jazz
#7. Metal 
#8. Pop
#9. Reggae
#10. Rock 

#There are various methods to do classification - Multiclass support vector machine, K-means 
#clustering, K-nearest neighbour, Convulational neural networks. 

#I will use K-nearest neighbour

#Feature extraction
#Our first step is to extract features from features and components from the audio files. Here, we 
#will identify the linguistic parts and discarding noise. 

#Mel frequency cepstral coefficients. 
#Generation of features - 
#1. Audio signals are constantly changing, first we divide these signals into smaller frames. 
#   Each frame is around 20-40 milliseconds long
#2. Then we try to identify different frequencies present in the frames. 
#3. Separate linguistic frequencies from the noise. 
#4. To discard the noise, it then takes discrete cosine transform (DCT) of these frequencies. Using 
#   DCT we will only keep a specific sequence of frequencies that have a high probability of information.

#import libraries 
from python_speech_features import mfcc
import scipy.io.wavfile as wav 
import numpy as np 

from tempfile import TemporaryFile
import os
import pickle
import random
import operator

import math

#define a function to get distance between the feature vectors and find neighbours 
def getNeighbour(trainingSet, instance, k):
    distance = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distance.append((trainingSet[x][2], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbour = []
    for x in range(k):
        neighbour.append(distance[x][0])
    return neighbour

#identify the nearest neighbour 
def nearestClass(neighbour):
    classVote = []

    for x in range(len(neighbour)):
        response = neighbour[x]
        if response in classVote:
            classVote[response]+=1
        else:
            classVote[response]=1
    
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

#define function for model evaluation 
def getAccuracy(testSet, predictions):
    correct = 0 
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return 1.0*correct/len(testSet)

#extract feature from dataset and dump features into binary .dat file 
directory = "__path_to_dataset__"
f= open("my.dat" ,'wb')
i=0
for folder in os.listdir(directory):
    i+=1
    if i==11 :
        break   
    for file in os.listdir(directory+folder):  
        (rate,sig) = wav.read(directory+folder+"/"+file)
        mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix , covariance , i)
        pickle.dump(feature , f)

f.close()

#train test split on the dataset 
dataset = []
def loadDataset(filename , split , trSet , teSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break  
    for x in range(len(dataset)):
        if random.random() <split :      
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])  
trainingSet = []
testSet = []
loadDataset("my.dat" , 0.66, trainingSet, testSet)

#Model training 
#make prediction using KNN and get accuracy 
leng = len(testSet)
predictions = []
for x in range (leng):
    predictions.append(nearestClass(getNeighbors(trainingSet ,testSet[x] , 5))) 
accuracy1 = getAccuracy(testSet , predictions)
print(accuracy1)
