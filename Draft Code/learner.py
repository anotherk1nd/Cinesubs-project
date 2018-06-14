#This is following the guide: https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-3

from preprocessing import *
from mfcc_class_method import *
import scipy as sp
import numpy as np
sp.random.seed(4321) #For reproducibility
np.random.seed(4321) #For reproducibility
from keras.layers import Dense, Input, LSTM, Conv1D, Conv2D, Dropout, Flatten, Activation, MaxPooling2D
from keras.models import Model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import pickle


def save_obj(obj, name ):
    # Saves to obj file I created in working directory
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    # Loads from obj file I created in working directory
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class learner(preprocessing, audio):
    #pass
    def __init__(self,subsfn,mp3,sr,duration=None):
        preprocessing.__init__(self,subsfn)
        audio.__init__(self,mp3,sr,duration)


    def len_check(self):
        # self.tconv(-1)
        # self.duration_sub = self.times[1] #gives time when last sub disappears off screen
        # self.duration_audio = lsa.get_duration(self.audio,sr=self.sr) #must have loaded mp3 first
        # if self.duration_sub > self.duration_audio:
        if len(self.mfccs) > len(self.pb_array):
            print("mfcc array longer than pb array")
            self.mfccs = self.mfccs[:len(self.pb_array),:]
        elif len(self.mfccs) < len(self.pb_array):
            print("pb_array longer than mfcc array")
            self.pb_array = self.pb_array[:len(self.mfccs),:]
        else:
            print("array lengths match")


    def train_test_splitting(self,test_size=0.33):
        X = self.mfccs
        y = self.pb_array[:,2]
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = self.test_size, random_state = 42)
        print(self.X_train.shape)

    # Conv-1D architecture. Just one sample as input
    def model_dense(self,input_shape):

        inp = Input(shape=input_shape)
        model = inp

        model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
        model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
        model = Flatten()(model)

        model = Dense(56)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = Dropout(0.2)(model)
        model = Dense(28)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        model = Model(inp, model)
        #fun2.model = model
        self.model = model

    def save_model(self,mfn):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(mfn, "w") as json_file:
            json_file.write(model_json)

    def save_weights(self,wfn):
        # serialize weights to HDF5
        model.save_weights(wfn)
        print("Saved model to disk")

# fun = audio('gotS07E01_16k.mp3',sr=16000,duration=3587)
# fun.load_mfcc("full_mfccs.npy")
# print(fun.mfccs)
# print(sp.shape(fun.mfccs))
# subs = preprocessing('gotS07E01.srt')
# subs.load_pb_array("pb_array.csv")

fun2 = learner('gotS07E01.srt','gotS07E01_16k.mp3',sr=16000,duration=None) #using duration=None should give full file. Pb array is filled based on last sub
#fun2.open_subs()
#fun2.audio_load()
#print(fun2.duration_true)
#fun2.audio_load(sr=fun2.sr)
#fun2.mfcc()
#fun2.save_mfcc("full_mfccs.npy")
fun2.load_mfcc("full_mfccs.npy")
print(fun2.mfccs)


fun2.load_pb_array("pb_array.npy")
fun2.len_check()
#fun2.train_test_splitting()

# rand = np.random.permutation(np.arange(len(fun2.y_train)))
# fun2.X_train = fun2.X_train[rand]
# fun2.y_train = fun2.y_train[rand]
#
# fun2.X_train = np.array([np.rot90(val) for val in fun2.X_train])
# fun2.X_train = fun2.X_train - np.mean(fun2.X_train, axis=0)
# print(fun2.X_train.shape, len(fun2.y_train[fun2.y_train==0]), len(fun2.y_train[fun2.y_train==1]), float(len(fun2.y_train[fun2.y_train==0]))/len(fun2.y_train[fun2.y_train==1]))
print(fun2.pb_array)
print(sp.shape(fun2.pb_array))
#print(fun2.pb_array[:,2])
#print(sp.shape(fun2.pb_array[:,2]))
print(sp.shape(fun2.mfccs))
fun2.mfccs = np.expand_dims(fun2.mfccs, axis=2) # reshape (569, 30) to (569, 30, 1)
print(sp.shape(fun2.mfccs))
#mccs_stack = sp.stack()
#input_shape = sp.shape(fun2.mfccs[1])
input_shape = (13,1) #for some reason, (13,) doesnt work, I think 1 here means 1 sample. sabater uses this too. Although Im not sure if i must rotate data
# print(input_shape)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=1, mode='min', patience=5)
checkpoint = ModelCheckpoint(filepath="checkpoint",monitor='val_loss', verbose=1, save_best_only=True)
history = History()
callbacks_list = [earlyStopping, checkpoint, history] #A callback is a set of functions to be applied at given stages of the training procedure.
fun2.model_dense(input_shape)


#fun2.model.load_weights("checkpoint")
fun2.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
fun2.train_test_splitting(test_size=0.33)
epoch_num = 2000
t0 = time.time()
hist = fun2.model.fit(fun2.X_train, fun2.y_train, epochs=epoch_num, batch_size=32, shuffle=True, validation_split=0.3, verbose=0, callbacks=callbacks_list)
t1 =time.time()
fun2.model.save_weights("Model Weights")
fun2.model.save("Model")
print(hist.history)
save_obj(hist.history,"hist")
#print('val_loss:', min(hist.history['val_loss']))
#print('val_acc:', max(hist.history['val_acc']))
scores = fun2.model.evaluate(fun2.X_test,fun2.y_test)
print(scores)
print("time: ",t1-t0)
prediction = fun2.model.predict(fun2.X_test)
print(prediction)
prediction_binary = sp.round_(prediction)
print(prediction_binary)
#print(np.where(prediction_binary!=fun2.y_test)) #Hangs on this for some reason
sp.save("Prediction.npy",prediction)
sp.save("Prediction Binary.npy",prediction_binary)
# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss   = hist.history['val_loss']
train_acc  = hist.history['acc']
val_acc    = hist.history['val_acc']
xc         = range(len(hist.history["val_loss"])) # number of epochs

plt.figure()
plt.plot(xc, train_loss, label="Training Loss")
plt.plot(xc, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training and Validation Loss on 1D CNN, 1 Hour of Audio")
#plt.savefig("train_val_loss.png")
plt.show()


#print(len(fun2.mfccs),len(fun2.pb_array))
#print(fun2.mfccs)
#fun2.load_pb_array("pb_array.npy")
#print(sp.shape(fun2.mfccs),sp.shape(fun2.pb_array))




# a = learner('gotS07E01.srt','gotS07E01_16k.mp3',16000,5)
# a.open_subs()
# a.pb_array_init()
# a.pb_array_fill()
# a.mfcc()
# print(a.mfccs)