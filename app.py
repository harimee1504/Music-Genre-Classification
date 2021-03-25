from flask import Flask,redirect,render_template,request
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa,librosa.display

from tempfile import TemporaryFile

import os
import pickle
import random
import shutil
import operator

import math

UPLOAD_FOLDER = 'static/data/'

app=Flask(__name__)

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


@app.route('/')
def index():
    try:
        shutil.rmtree('static/data')
    except:
        os.makedirs('static/data')        
    os.makedirs('static/data')
    return render_template("index.html")

@app.route('/upload',methods=['GET','POST'])
def main():
    if request.method=='POST':      
        file = request.files['file']
        filename = "test.wav"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        file="static/data/test.wav"

        #audio signal
        signal,sr=librosa.load(file,sr=22050)
        librosa.display.waveplot(signal,sr=sr)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title('Signal')
        a="static/data/"+str(random.randint(215500,352400))+'a'
        plt.savefig(a)
        plt.close()

        #Fast Fourier Transform
        fft=np.fft.fft(signal)

        magnitude=np.abs(fft)
        frequency=np.linspace(0,sr,len(magnitude))

        left_frequency=frequency[:int(len(frequency)/2)]
        left_magnitude=magnitude[:int(len(frequency)/2)]

        plt.plot(left_frequency,left_magnitude)
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('Fast Fourier Transform')
        b="static/data/"+str(random.randint(215500,352400))+'b'
        plt.savefig(b)
        plt.close()

        #short fourier transform 

        n_fft=2048
        hop_length=512

        stft=librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)
        spectrogram = np.abs(stft)

        #log_spectrogram=librosa.amplitude_to_db(spectrogram)

        librosa.display.specshow(spectrogram,sr=sr,hop_length=hop_length)

        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.title('Short Fourier Transform')
        c="static/data/"+str(random.randint(215500,352400))+'c'
        plt.savefig(c)
        plt.close()

        #MFCC

        MFCCs=librosa.feature.mfcc(signal,n_fft=n_fft,hop_length=hop_length,n_mfcc=13)

        librosa.display.specshow(MFCCs,sr=sr,hop_length=hop_length)

        plt.xlabel("Time")
        plt.ylabel("MFCC")
        plt.colorbar()
        plt.title('Mel-frequency cepstral coefficients (MFCCs)')
        d="static/data/"+str(random.randint(215500,352400))+'d'
        plt.savefig(d)
        plt.close()

        # function to get the distance between feature vecotrs and find neighbors
        def getNeighbors(trainingSet, instance, k):
            distances = []
            for x in range(len(trainingSet)):
                dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
                distances.append((trainingSet[x][2], dist))

            distances.sort(key=operator.itemgetter(1))
            neighbors = []
            for x in range(k):
                neighbors.append(distances[x][0])
            
            return neighbors

        # identify the class of the instance
        def nearestClass(neighbors):
            classVote = {}

            for x in range(len(neighbors)):
                response = neighbors[x]
                if response in classVote:
                    classVote[response] += 1
                else:
                    classVote[response] = 1

            sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)

            return sorter[0][0]

        # function to evaluate the model
        def getAccuracy(testSet, prediction):
            correct = 0
            for x in range(len(testSet)):
                if testSet[x][-1] == predictions[x]:
                    correct += 1
            
            return (1.0 * correct) / len(testSet)

        # Split the dataset into training and testing sets respectively
        dataset = []

        def loadDataset(filename, split, trSet, teSet):
            with open('static/TrainedModel/my.dat', 'rb') as f:
                while True:
                    try:
                        dataset.append(pickle.load(f))
                    except EOFError:
                        f.close()
                        break
            for x in range(len(dataset)):
                if random.random() < split:
                    trSet.append(dataset[x])
                else:
                    teSet.append(dataset[x])

        trainingSet = []
        testSet = []
        loadDataset('static/TrainedModel/my.dat', 0.66, trainingSet, testSet)
        
        def distance(instance1 , instance2 , k ):
            distance =0 
            mm1 = instance1[0] 
            cm1 = instance1[1]
            mm2 = instance2[0]
            cm2 = instance2[1]
            distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
            distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
            distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
            distance-= k
            return distance

        # making predictions using KNN
        leng = len(testSet)
        predictions = []
        for x in range(leng):
            predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

        accuracy1 = getAccuracy(testSet, predictions)
        #print(accuracy1)

        test_dir = "static/data/"
       
        test_file = test_dir + "test.wav"
       
        (rate, sig) = wav.read(test_file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        
        feature = (mean_matrix, covariance, 0)

        results={1: 'blues', 2: 'classical', 3: 'country', 4: 'disco', 5: 'hiphop', 6: 'jazz', 7: 'metal', 8: 'pop', 9: 'reggae', 10: 'rock'}

        pred = nearestClass(getNeighbors(dataset, feature, 5))
        #print(results[pred])
        img='static/images/'+results[pred]+'.jpeg'

        return render_template("result.html",resultimg=img,result=results[pred].upper(),accuracy=round((accuracy1*100),4),a=a+".png",b=b+".png",c=c+".png",d=d+".png")
    
    else:
        return redirect('/')


if __name__ == '__main__':
    app.run(debug=True, host='localhost',port=80)
    