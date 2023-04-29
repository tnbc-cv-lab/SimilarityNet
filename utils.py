import matplotlib as plt




def plot_accuracy(history):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = './model_accuracy.png'
    plt.savefig(plot_path)