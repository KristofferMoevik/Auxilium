#!/usr/bin/env python3

import socket
import pandas as pd
import cv2

HOST = '10.22.27.154'  # Standard loopback interface address (localhost)
PORT = 65000        # Port to listen on (non-privileged ports are > 1023)

def emotion_list_to_sorteddict(emotion_vector): #return a sorted dict of emotion values
                feelingslst = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad","Suprised"]

                #sortere følelser
                feelingsdic = {}
                n=0
                for i in emotion_vector:
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

def return_focus(visualattention, blinksperminute):
                #vekting av øynene 
                k1 = 3 #visualattention
                k2 = 2 #blinksprminute

                #kalkuleree focus 
                if(visualattention == 0):
                    focus = 0
                elif(blinksperminute > 30):
                    focus = (100/(k1+k2))*((k1*visualattention + 0))
                elif(blinksperminute < 2):
                    focus = (100/(k1+k2))*((k1)*visualattention + k2)
                else:
                    focus = (100/(k1+k2))*(k1*visualattention + k2*(-1/(28)*blinksperminute + 15/14))
                return focus



wall_margin = 10
inital_margin = 30
max_width = 300
text_bar_margin = 10
element_margin = 80
bar_height = 30

#if img is not None:
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            rec=bytes.decode(data,'utf-8')
            blink_rate_rcv,emotion_rcv,focus_rcv=rec.split(';')
            #print("Blink rate:", blink_rate_rcv)
            print("Emotion: ",emotion_rcv)
            #print("Focused: ",focus_rcv)

            #emotion data handling
            emotion_string = emotion_rcv[2:-1]
            emotion_string_vector = list(emotion_string.split(", "))
            emotion_vector = []
            for i in emotion_string_vector:
                emotion_vector.append(float(i))
                
            focus = return_focus(float(focus_rcv), float(blink_rate_rcv))
            print(focus)
            sorted_emotion_dict = emotion_list_to_sorteddict(emotion_vector)
            print(sorted_emotion_dict)
                
            emotion_key_list = list(sorted_emotion_dict)

            img = cv2.imread("white.jpg")
            img = cv2.resize(img, (600, 400))

            cv2.putText(img,"Focus",(wall_margin, inital_margin),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
            cv2.rectangle(img, (wall_margin, inital_margin + text_bar_margin), (wall_margin + int(max_width*focus/100), inital_margin + text_bar_margin + bar_height), (255, 0, 0), -1)

                
            cv2.putText(img, emotion_key_list[0], (wall_margin, inital_margin+element_margin), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
            cv2.rectangle(img, (wall_margin, inital_margin + element_margin + text_bar_margin), (wall_margin + int(max_width*(sorted_emotion_dict.get(emotion_key_list[0])/100)), inital_margin + element_margin + text_bar_margin + bar_height), (255, 0, 0), -1)
                
            cv2.putText(img,emotion_key_list[1],(wall_margin, inital_margin+2*element_margin),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
            cv2.rectangle(img, (wall_margin, inital_margin + 2*element_margin + text_bar_margin), (wall_margin + int(max_width*(sorted_emotion_dict.get(emotion_key_list[1])/100)), inital_margin + 2*element_margin + text_bar_margin + bar_height), (255, 0, 0), -1)

            cv2.putText(img,emotion_key_list[2],(wall_margin, inital_margin+3*element_margin),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
            cv2.rectangle(img, (wall_margin, inital_margin + 3*element_margin + text_bar_margin), (wall_margin + int(max_width*(sorted_emotion_dict.get(emotion_key_list[2])/100)), inital_margin + 3*element_margin + text_bar_margin + bar_height), (255, 0, 0), -1)
                
            cv2.imshow("gui", img)
            cv2.waitKey(1)

#else:
#    print("Image is none")