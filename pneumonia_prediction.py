# -*- coding: utf-8 -*-
"""
@author: Herikc Brecher
Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""

'''
Definition of the Business Problem:
        
    What is pneumonia?
        Pneumonia is an inflammatory condition of the lung that mainly affects the small air sacs known as alveoli.
    Symptoms usually include some combination of a productive or dry cough, chest pain, fever, and difficulty breathing.
    The severity of the condition is variable. Pneumonia is usually caused by infection by viruses or bacteria and less commonly by
    other microorganisms, certain medications or conditions such as autoimmune diseases. Risk factors include cystic fibrosis,
    chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, poor ability to cough, as after a stroke
    and a weak immune system. The diagnosis is usually based on symptoms and physical examination. Chest radiography,
    blood and sputum culture can help confirm the diagnosis. The disease can be classified according to the place where it was
    acquired, such as community-acquired or hospital-acquired pneumonia or associated with healthcare.
    
        The business problem is dealing with lives, where a wrong prediction could be costing someone else's life. The goal is
    bring the highest possible precision to the model, analyzing models tirelessly until we create the best model possible. The goal
    in costume, it is obviously 100%, lives are priceless. However, Deep Learning is math and mistakes can happen, with that in mind.
    A target of at least 97% accuracy is set, with an ideal target of 98.5%.
        
        As input the model will receive an x-ray of the chest, where it will classify whether that person has pneumonia or not.
        
    Suggestions for Improvement:
        1 - Use Transfer-Learning (Insufficient Memory)
        2 - Use Tuning to find the best parameters (Using already in hyperparameters)
        3 - Use of larger images (Images above 248 make tests impossible)
        4 - Apply a new split, as it has a large imbalance between the datasets (Applied, observed significant improvement)
        5 - Pre-apply image filters
        6 - Reduce batch size (Batch size of 16 generated a gain in accuracy, but an increase in Loss.)
            
        Note: For items 1 and 3, a computer with high availability by ram memory is required.
'''

# Import from libraries

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import type_models.utils as u

####### FUNCTIONS #######

# Function for reading images and applying filters
# Opencv grayscale applied and resized to IMG_SIZE
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            # Grayscale and image reshape
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            
            resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)) # COMMENT FOR EXPLORATORY ANALYSIS
            
            data.append([resized_arr, class_num])
    return np.array(data)

# Add row to dataframe
def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1  
    return df.sort_index()    

####### DEFINITION OF GLOBAL VARIABLES FOR ENVIRONMENTAL EXECUTION #######

labels = ['PNEUMONIA', 'NORMAL']
IMG_SIZE = 150
BATCH_SIZE = 16
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

'''
    After checking the average size of the images, the following statement arrives:
        max_height = 2916
        max_width = 2713
        media_height = 1327
        media_width = 970
        min_height = 384
        min_width = 127
    However, viewing through a histogram, it is stated that the mean is not representative,
    thus, several tests were carried out, until the IMG_SIZE was reached between 128 and 250.
    After analyzing some cases, 150 were adopted, since 250 were at risk of decreasing
    accuracy without increasing the complexity of the model.
'''

train = get_data('Dataset/chest_xray/train')
test = get_data('Dataset/chest_xray/test')
val = get_data('Dataset/chest_xray/val')

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

# The dataset is separated into vectors with the images and labels, according to its definition
for array_image, label in train:
    x_train.append(array_image)
    y_train.append(label)

for array_image, label in test:
    x_test.append(array_image)
    y_test.append(label)
    
for array_image, label in val:
    x_val.append(array_image)
    y_val.append(label)

####### EXPLORATORY ANALYSIS #######

'''
    Analyzing the graph below, it can be seen that the dataset is totally unbalanced,
    has many more images of Pneumonia, for better performance it is necessary
    apply a new distribution on the dataset
'''
l = []
for i in y_train:
    if(i == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.countplot(l) 
plt.savefig('count_normal_pneumo.png', format='png')       
 
'''
    For better analysis, all exploratory analysis coding was separated in the utils.py file,
    it is recommended to read the utils.py file to understand the results and what they mean.
    
    Brief analysis:
        The main analysis was performed on the measures of width and height, this evaluation was carried out,
    in order to reach an IMG_SIZE with greater representativeness. It was evaluated by the Histogram,
    and Quartiles where there should be a greater focus. The box plot served as an analysis, to identify,
    in general the view of the dataset.
        A big problem with the analysis is that the informed outputs brought results that were difficult to simulate,
    arriving at times when an implementation would become unviable or exhaustive.
        The suggestion for training more complex models is to rent Clusters from Google.
'''

#x_total = x_train + x_val + x_test

# Performed the analysis for each split, after pre-processing can still be performed again for a new comparison

'''
u.main(x_train)
u.main(x_val)
u.main(x_test)
'''

####### PRE-PROCESSING #######

'''
    After analyzing the inconsistency in the dataset, it is analyzed that a new split can help in the performance.
    The performance had an average increase of 5% in the analyzes performed.
'''

### NEW SPLIT ###

# Joining all the data in the old list format in a single list of total x and y
x_total = x_train + x_val + x_test
y_total = y_train + y_val + y_test

# Converting the total list to data frame and filling in
total_df = pd.DataFrame(columns = ['Img', 'Label'])

for i in range(len(x_total)):
    add_row(total_df, [x_total[i], y_total[i]])
   
# Separating into training and test data
x_train, x_test, y_train, y_test = train_test_split(total_df.drop('Label',axis=1),
                                                    total_df['Label'],                             
                                                    test_size = 0.3)

# Separating the data again between testing and validation
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.05)

# Dataframe to list conversion performed again
x_train = x_train['Img'].values.tolist()
x_val = x_val['Img'].values.tolist()
x_test = x_test['Img'].values.tolist()

y_train = y_train.values.tolist()
y_val = y_val.values.tolist()
y_test = y_test.values.tolist()



# Pixel normalization is performed for faster execution
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# Resizing is applied over the array, to a format that Keras accepts
x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)


# Augmentation is used to have an outlier reduction and greater precision
datagen = ImageDataGenerator(rotation_range = 30, zoom_range = 0.2, width_shift_range = 0.1, height_shift_range = 0.1, 
                             horizontal_flip = True)
datagen.fit(x_train)



# The Convolutional Neural Network is created
'''
    The neural network below was created in a block structure:
    Conv2D - Each block has a Dense layer, to recalculate the weights from filters, to each layer
    more filters are applied, so we can have more precision in the pixels.
    
    BatchNormalization - After the new calculations, a new normalization of the output values ​​is performed,
    so we maintain efficiency in calculations and improving accuracy.
    
    MaxPooling2D - Used to prioritize the highest output values, as these have the greatest impact.
    
    Dropout - To avoid the outlier, Dropout is applied where values ​​that differ
    many of the others are discarded.
'''

model = Sequential()

# Primeiro bloco (Entrada)
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = INPUT_SHAPE))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Second Block
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Third Block
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Fourth Block
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Fifth Blocks
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Sixth Block - Flatten is used to convert pixels from matrix to array
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))

# Output
model.add(Dense(units = 1 , activation = 'sigmoid'))


# Build using RMSPROP or ADAM
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()


# A callback is created to reduce the training rate and we don't fall in minimum places
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1, factor=0.3, min_lr = 0.000001)



# The model is trained
model_history = model.fit(datagen.flow(x_train,y_train, batch_size = BATCH_SIZE), epochs = 13,
                    validation_data = datagen.flow(x_val, y_val), callbacks = [learning_rate_reduction])



# The model is applied on the test basis to measure its accuracy
print("Loss: " , model.evaluate(x_test,y_test)[0])
print("Accuracy: " , model.evaluate(x_test,y_test)[1]*100 , "%")

# Analysis of the evolution of accuracy throughout the times
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

# Analysis of loss over time
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

# The prediction value is stored
predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1,-1)[0]

'''
    The data about the model is displayed:
        
    Accuracy: The hit rate minus error, placed in%
    Recall: The rate of hit frequency, how much do you hit each error?
    f1-score: F1 is the combination of precision and recall in a single score.
'''

print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

# The precision matrix is ​​created to visualize the number of correct cases
cm = confusion_matrix(y_test,predictions)
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])   
 
plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)



# The model is saved its shape and weights, for future use
model_json = model.to_json()
model_name = 'classifier'
with open("models/complexity_model_advanced/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/complexity_model_advanced/" + model_name + ".h5")

















