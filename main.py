import numpy as np
import cv2
import skimage
import keras.api._v2.keras as keras

'''
output:
letter: the recognized letter from the given picture
confidence: the possibility of correction

input params:
filePath: full path of image file
imgSize: this should be same as imageSize which is used by model. It is 32.
model: a CNN model. It should be created and trained when the website start, then saved in a global variable 
'''
def load_img(filePath, imgSize, model):
    dictLabels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                  'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                  'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
    X = np.empty((1, imgSize, imgSize, 3), dtype=np.float32)
    img_file = cv2.imread(filePath)
    if img_file is not None:
        img_file = skimage.transform.resize(img_file, (imgSize, imgSize, 3))
        img_arr = np.asarray(img_file).reshape((-1, imgSize, imgSize, 3))
        X[0] = img_arr

    prediction = model.predict(X)
    letterNum = prediction[0].tolist().index(max(prediction[0]))

    # convert num to letter
    letter = None
    keys = [k for k, v in dictLabels.items() if v == letterNum]
    if keys:
        letter = keys[0]

    confidencePercentage = max(prediction[0])
    return letter, confidencePercentage



if __name__ == '__main__':
    '''
    Sample code to create and train model
    To shorten training time, you can set (but it is not accurate at all)
    maxFileNumPerFolder = 1
    imageSize = 64
    epochs = 1
    
    To get accurate result (>50% correction), you can set
    maxFileNumPerFolder = 100
    imageSize = 64
    epochs = 10
    '''

    imageSize = 64

    # load model
    model_path = "C:\\GitHub-Yang\\ASLApi\\model"
    model = keras.models.load_model(model_path)


    # sample code to load image and predict it with confidence
    filePath = "C:\\GitHub-Yang\\F.jpg"
    predictLetter, confidence = load_img(filePath, imageSize, model)
    print(predictLetter + ": " + str(confidence * 100) + "%")
