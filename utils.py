import matplotlib.pyplot as plt


RESULT_PATH = '/home/niranjan.rajesh_ug23/TNBC/SimilarityNet/Results/'

def plot_accuracy(history):
    plt.clf()
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = RESULT_PATH+'model_acc.png'
    plt.savefig(plot_path)