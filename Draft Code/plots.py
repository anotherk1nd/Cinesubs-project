import pickle
import matplotlib.pyplot as plt
plt.close("all")

def load_obj(name ):
    # Loads from obj file I created in working directory
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

history = load_obj("hist")
print(history)
print(len(history["val_loss"]))

# visualizing losses and accuracy
train_loss = history['loss']
val_loss   = history['val_loss']
train_acc  = history['acc']
val_acc    = history['val_acc']
xc         = range(len(history["val_loss"])) # number of epochs

plt.figure()
plt.plot(xc, train_loss, label="Training Loss")
plt.plot(xc, val_loss, label="Validation Loss")
#plt.plot(xc, train_acc, label="Training Accuracy")
#plt.plot(xc, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training and Validation Loss on 1D CNN, 1 Hour of Audio")
plt.legend()
plt.savefig("train_val_loss.png")
plt.show()

