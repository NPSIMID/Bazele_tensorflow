# Functia de vizualizare aliatoare a imaginilor

# Importul bibliotecilor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf

def view_random_image(target_dir, target_class):
  # fixarea directoriului 
  target_folder = target_dir+target_class

  # creara unui cai aliatoare
  random_image = random.sample(os.listdir(target_folder), 1)

  # citirea imaginii si afisarea ei
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # afisarea formei imaginii

  return img

# Functia de afisare separata a curbei pierderilor si a acuratetei modelului
def plot_loss_curves(history):
  """
  Returneaza curba pierderilor si acuraterea in grafice diferite.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Afisarea pierderilor
  plt.plot(epochs, loss, label="Pierderile de training")
  plt.plot(epochs, val_loss, label='Pierderile de test')
  plt.title('Pierderile')
  plt.xlabel('Epochs')
  plt.legend()

  # afisarea acuratetei
  plt.figure()
  plt.plot(epochs, accuracy, label="Acuratetea de training")
  plt.plot(epochs, val_accuracy, label='Acuratetea de test')
  plt.title('Acuratetea')
  plt.xlabel('Epochs')
  plt.legend();

# Functia ce importa imaginea si o redimensioneaza pentru a putea fi aplicata modelului
def load_and_prep_image(filename, img_shape=224):
  """
  Citeste imaginea din filename, o transforma intr-un tensor 
  si o redimensioneaza in forma(img_shape, img_shape, colour_channel).
  """
  # citirea imaginii din directoriu
  img = tf.io.read_file(filename)

  # Decodarea imagini citite si foramrea unui tensor asigurandu-se 3 canale a culorilor 
  img = tf.image.decode_image(img, channels=3)

  # Redimensionarea imaginii ( in aceleasi dimensiune ca imaginile de training a modelului)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Normarea valoriulor pixelor imaginii
  img = img/255.
  
  return img

# Functia de incarcare a imaginii, realizare a predictiei si afisare a rezultatelor
def pred_and_plot(model, filename, class_names):
  """
  Importul imaginii localizate in filename, realizarea predictie cu model si
  afisarea imginii cu titlul drep rezultat al predictiei.
  """
  # Importul imaginii si procesare ei
  img = load_and_prep_image(filename)

  # adaugarea unei dimendiuni suplimentarea
  img_dim=tf.expand_dims(img, axis=0)

  # Realizarea predictie
  pred = model.predict(img_dim)
  print(pred)

  # adaugarea unei logici ce va permite utilizarea functie petru calsificare binara si clasificare categorica
  #verificarea daca predictia area mai mult de o valoarea (clasificare categorica)
  if len(pred[0])>1:
    # se alege denumirea clasei predictia cu indexul maxim
    pred_class = class_names[tf.argmax(pred[0])]
  # caca predictia are doar o valoarea (clasificare binara)
  else: 
    # Obtinerea denumirii clasei
    pred_class = class_names[int(tf.round(pred))]

  # Afisarea imaginii si a clasei prezise
  plt.imshow(img)
  plt.title(f"Predictie: {pred_class}")
  plt.axis(False);

# Functia de crearea a unui callback TensorBoard 
import datetime
import tensorflow as tf
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir )

  print(f"Fisierele de inregistrarea TensorBoard se salveaza in : {log_dir}")
  return tensorboard_callback

# Functia de crearea a modelelor cu extragerea caracteristicilor
def create_model(model_url, num_classes=10):
  """
  Creaza un model Keras Sequential dupa url-ul TensorFlow Hub
  """
  # descarcarea modelului si salvarea sa ca un nivel Keras
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable = False, # se ingheata nivelele
                                           name = 'Nivelul_extragere_caracteristici', # numele nivelului pentru analiza
                                           input_shape = IMAGE_SHAPE+(3,)) # definirea formei intrarilor
  #crearea modelului propriu
  model = tf.keras.Sequential([
                               feature_extractor_layer, # utilizarea nivelului importat din tensorFlow Hub
                               layers.Dense(num_classes, activation='softmax', name = "Nivel_de_iesire") # crearea nivelului propriu de iesire
  ])

  # compilarea modelului propriu
  model.compile(loss = "categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ['accuracy'])
  return model

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# Create function to unzip a zipfile into current working directory 
# (since we're going to be downloading and unzipping a few files)
import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory.
  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results