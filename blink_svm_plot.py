import pandas as pd
import cv2
import numpy as np
import copy
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import argparse
import time
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


DATA = pd.read_csv('output_SVM_model_2.csv')
################################################
FRAME_LIST = list(DATA.index)
BLINK_LIST = list(DATA.blink)

#sostituisco 0.0 o 1.0 sparsi
for n in range(len(BLINK_LIST)):
    #trovo il primo 1.0
    if BLINK_LIST[n]==1.0:
        i = copy.deepcopy(n)
        #correggi 1.0 isolati: se è un 1.0 singolo (o doppio) diventa 0.0 (o 0.0 0.0)
        if sum(BLINK_LIST[i:i+6])<3.0:
            BLINK_LIST[i]=0.0
        else:
            #correggi 0.0 isolati: se ci sono 0.0 singoli (o doppi) (o tripli) diventano 1.0 (o 1.0 1.0) (o 1.0 1.0 1.0)
            while (sum(BLINK_LIST[i:i+6])>=3.0):
                BLINK_LIST[i+1]=1.0
                BLINK_LIST[i+2]=1.0
                i+=1

#ora costruisco singoli 1.0 corrispondenti al blink
for n in range(len(BLINK_LIST)):
    #trovo il primo 1.0
    if BLINK_LIST[n]==1.0:
        i = copy.deepcopy(n)
        while (BLINK_LIST[i+1]==1.0):
            BLINK_LIST[i+1]=0.0
            i+=1

#scala gli 1.0 di 5 frame per posizionarlo alla chiusura circa
BLINK_LIST=[0.0,0.0,0.0,0.0,0.0]+BLINK_LIST[:len(BLINK_LIST)-5]


BLINK_LIST = pd.DataFrame(BLINK_LIST, index=FRAME_LIST)
BLINK_LIST.index.name='frame'
BLINK_LIST.columns = ['blink']

#########################################################################################
########################################################################################
#unisco blink_ajust con video_shocase
######################################################################################
###################################################################################
result=BLINK_LIST
######################################################################

our_video=cv2.VideoCapture(args["video"])
fps=our_video.get(cv2.CAP_PROP_FPS)
fps=int(fps)
print("video a {} fps". format(fps))
# start the video stream thread
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

raw_data=pd.read_csv("tmp.csv", index_col="frame")

dati=raw_data
listear=list(dati.ear)
#normalizzo
listear=np.array(listear)
listear=(listear-np.nanmin(listear))/(np.nanmax(listear)-np.nanmin(listear))
listear=list(listear)
LIST_EAR_PER_TABELLA_PREVISIONI=listear
LIST_EAR_PER_TABELLA_PREVISIONI=pd.Series(LIST_EAR_PER_TABELLA_PREVISIONI, index=range(0,len(LIST_EAR_PER_TABELLA_PREVISIONI)))

raw_data_1=raw_data.threshold
SHOWCASE_DATA=pd.concat([raw_data_1, result,LIST_EAR_PER_TABELLA_PREVISIONI], axis=1 )
SHOWCASE_DATA=SHOWCASE_DATA.fillna(0)
SHOWCASE_DATA.columns=["threshold","blink","ear_norm"]




SHOWCASE_DATA_CUMSUM=SHOWCASE_DATA.cumsum(axis=0)
SHOWCASE_DATA_CUMSUM=SHOWCASE_DATA_CUMSUM.drop('ear_norm', 1)

def mediaMoblieBlinkRate(df, wind):
    list_blink=list(df)
    list_blink_tmp=list()
    for i in range(wind,len(list_blink)):
        list_blink_tmp.append(sum(list_blink[i-wind:i])/wind)
    series_blink=pd.Series(list_blink_tmp, index=range(wind,len(list_blink)))
    return series_blink

def smoth_BR_moving_av(df, wind):
    indici=df.index
    list_blink=list(df)
    list_blink_tmp=list()
    for i in range(int(wind/2),len(list_blink)-int(wind/2)):
        list_blink_tmp.append(sum(list_blink[i-int(wind/2):i+int(wind/2)])/wind)
    series_blink=pd.Series(list_blink_tmp, index=range(indici[0]+int(wind/2),indici[-1]-int(wind/2)))
    return series_blink

DF_BLINK=SHOWCASE_DATA


#calcolo medie mobili per fps con finestre da 20 sec
DF_MOV_BR=mediaMoblieBlinkRate(list(DF_BLINK.blink), 20*fps)
#trasformo blink per frame in blink per min
DF_MOV_BR=DF_MOV_BR*fps*60
SMOOTH_BR=DF_MOV_BR.rolling(window=3*fps,center=False).mean()

DF_BLINK=DF_BLINK[DF_BLINK.blink>0]

FRAME = 1

secondi=SHOWCASE_DATA.index/fps
my_xticks=list()

for i in secondi:
    my_xticks.append(time.strftime("%M:%S", time.gmtime(i)))

SMOOTH_BR.plot(color="r", label="Blink/min")
plt.legend(loc=2, prop={'size': 8})
plt.axvline(x=FRAME, color="black")
frequency=fps*30
plt.xticks(SHOWCASE_DATA.index[::frequency], my_xticks[::frequency], rotation=45)
plt.savefig('plot.png')
plt.close()

cv2.destroyAllWindows()
vs.stop()