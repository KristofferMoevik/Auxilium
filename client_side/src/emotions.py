import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import data_generator
from gaze_tracking import GazeTracking
import time
import socket

gaze = GazeTracking()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')

# validation_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

a=0
# If you want to train the same model or try other models, go for this
#if mode == "train":
#    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
#    model_info = model.fit_generator(
#            train_generator,
#            steps_per_epoch=num_train // batch_size,
#            epochs=num_epoch,
#            validation_data=validation_generator,
#            validation_steps=num_val // batch_size)
#    plot_model_history(model_info)
#    model.save_weights('model.h5')

# emotions will be displayed on your face from the webcam feed

def emotion_list_to_sorteddict(emotion_vector): #return a sorted dict of emotion values
    feelingslst = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad","Suprised"]

    #sortere følelser
    feelingsdic = {}
    n=0
    for i in emotion_vector[0]:
        feelingsdic[feelingslst[n]] = i*100
        n+=1
        #print(feelingsdic)

    sortedfeelingslst = sorted(feelingsdic.items(),key=lambda x:x[1],reverse=True)

    #print(sortedfeelingslst)
    #print(sortedfeelingslst[0][1])
    sortedfeelingsdic = {}
    for element in sortedfeelingslst:
        sortedfeelingsdic[element[0]] = element[1]

    return sortedfeelingsdic

def return_focus(average_center_view_last_min, blinksperminute):
    #vekting av øynene 
    k1 = 3 #average_center_view_last_min
    k2 = 2 #blinksprminute

    #kalkuleree focus 
    if(average_center_view_last_min == 0):
        focus = 0
    elif(blinksperminute > 30):
        focus = (100/(k1+k2))*((k1*average_center_view_last_min + 0))
    elif(blinksperminute < 2):
        focus = (100/(k1+k2))*((k1)*average_center_view_last_min + k2)
    else:
        focus = (100/(k1+k2))*(k1*average_center_view_last_min + k2*(-1/(28)*blinksperminute + 15/14))
    return focus


if mode == "display":
    model.load_weights('client_side//src//model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    emotion_list = []
    # start the webcam feed
    
    cap = cv2.VideoCapture(0)

    blinks = 0
    been_closed = False
    blinks_per_min = 0
    blinks_list_last_min = []
    average_blinks_per_min = 0
    average_blinks_per_min_list = []

    view_center = 0
    view_center_list_last_sec = []
    average_center_view_last_sec = 0
    center_view_list_min = []
    average_center_view_last_min = 0 
    center_view_last_min_list = []

    start_time = time.time()

    HOST = '10.22.229.65'  # The server's hostname or IP address
    PORT = 65000        # The port used by the server
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    start_time_TCP=time.time()
    while True:
        end_time = time.time()
        time_lapsed = end_time - start_time

        if time_lapsed >= 1:
            start_time = time.time()
            if blinks >= 1:
                blinks_list_last_min.append(blinks/time_lapsed)
                blinks = 0
                if len(blinks_list_last_min) >= 60:
                    blinks_list_last_min.pop()
            average_blinks_per_min = sum(blinks_list_last_min)     
            
            if len(view_center_list_last_sec) != 0:
                a = (sum(view_center_list_last_sec) / len(view_center_list_last_sec))/ time_lapsed
                if ((sum(view_center_list_last_sec) / len(view_center_list_last_sec))/ time_lapsed) > 0:
                    average_center_view_last_sec = (sum(view_center_list_last_sec) / len(view_center_list_last_sec))/ time_lapsed
                
                view_center_list_last_sec = []
                center_view_last_min_list.append(average_center_view_last_sec)
                if len(center_view_last_min_list) >= 60:
                    center_view_last_min_list.pop(0)
                average_center_view_last_min = sum(center_view_last_min_list) / len(center_view_last_min_list)

        ret, frame = cap.read()
        text = ""
        gaze.refresh(frame)
        frame=gaze.annotated_frame()
        if gaze.is_blinking():
            been_closed = True
        if(gaze.is_center()):
            cv2.putText(frame, "Attention: True", (90, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (147, 58, 31), 1)
            view_center_list_last_sec.append(1)
            
        else:
            cv2.putText(frame, "Attention: False", (90, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (147, 58, 31), 1)
            view_center_list_last_sec.append(0)
        if(not gaze.is_blinking() and been_closed):
            been_closed = False
            blinks += 1
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"
        
        cv2.putText(frame,"Focus this min: "+str(average_center_view_last_min),(100,400),cv2.FONT_HERSHEY_COMPLEX,1,(147, 58, 31),1)
        cv2.putText(frame,"Blinks: "+str(average_blinks_per_min),(400,450),cv2.FONT_HERSHEY_COMPLEX,1,(147, 58, 31),1)
        cv2.putText(frame, text, (90, 350), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        #cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('client_side//src//haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for iteration, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            #write data
            average_blinks_per_min_list.append(average_blinks_per_min)
            #emotion_list.append(prediction.tolist())
            #sorted_emotion_dict = emotion_list_to_sorteddict(prediction.tolist())
            #print(sorted_emotion_dict)
            #focus = return_focus(average_center_view_last_min, average_blinks_per_min)
            #print(focus)
            #print("/n")
            
            if((time.time()-start_time_TCP)>1):
                start_time_TCP=time.time()
                
                
                message=str(average_blinks_per_min)+str('; ')+str(prediction.tolist()[0])+str('; ')+str(average_center_view_last_min)
                s.send(bytes(message,'utf-8'))
                

        cv2.imshow('Video', cv2.resize(frame,(1000,600),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #data_generator(emotion_list, blinks_per_min_list, view_list)
    cv2.destroyAllWindows()