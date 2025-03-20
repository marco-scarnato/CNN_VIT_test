import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Separate the dataset dir into training, validation and testing sets
# The division is already done in the dataset dir.
#   Input: path to the dataset dir
#   Output: path to the training dir, validation dir and testing dir 
def dataset_sub_division(path):
    train_dir = path + "\\train"
    valid_dir = path + "\\valid"
    test_dir = path + "\\test"
    return train_dir, valid_dir, test_dir

# Display the composition of the dataset
#   Input: path to the dataset dir
#   Output: None
# Display the number of classes and the number of images of the dataset
def analisys_dataset_composition(path):

    # Display the number of classes in the dataset
    dataset_classes = os.listdir(path)
    print("\nTotal number of classes are: ", len(dataset_classes))

    # Display the number of images of the dataset
    print("\nTotal number of images in the dataset are: ", 
          sum([len(os.listdir(path + "\\" + i)) for i in dataset_classes]))
    
    print("MEAN number of images in each class is: ", 
          round(sum([len(os.listdir(path + "\\" + i)) for i in dataset_classes]) / len(dataset_classes)))
    
    len_classes = [len(os.listdir(path + "\\" + i)) for i in dataset_classes]
    name_classes = [i for i in dataset_classes]

    # Display the number of images for each class
    print("\nNumber of images for each class: ")
    fig, ax = plt.subplots(figsize=(10, 10))
    # Creazione del barplot
    len_classes.sort()
    sns.barplot(x=len_classes, y=name_classes, ax=ax)
    plt.show()
    return name_classes

# Display the composition of the dataset
#   Input: path to the dataset dir
#   Output: None
# Show an image for every class we are working with.
def show_dataset_composition(path, images_per_row=4, img_size=(3, 3)):
    dataset_classes = os.listdir(path)
    num_classes = len(dataset_classes)

    # Calcola quante righe servono
    rows = (num_classes // images_per_row) + (num_classes % images_per_row > 0)
    
    # Crea la figura con sottoplot
    fig, axes = plt.subplots(rows, images_per_row, figsize=(img_size[0] * images_per_row, img_size[1] * rows))
    
    # Se c'è solo una riga, assicuriamoci che 'axes' sia una lista
    if rows == 1:
        axes = [axes]

    for i, cl in enumerate(dataset_classes):
        row, col = divmod(i, images_per_row)  # Determina posizione nella griglia
        image_example_path = os.listdir(os.path.join(path, cl))[0]
        image = plt.imread(os.path.join(path, cl, image_example_path))

        ax = axes[row][col] if rows > 1 else axes[col]  # Se una sola riga, non usare [row]
        ax.imshow(image)
        ax.set_title(cl, fontsize=10)
        ax.axis("off")  # Nasconde gli assi

    # Nasconde gli assi vuoti se ci sono meno immagini delle celle disponibili
    for i in range(num_classes, rows * images_per_row):
        row, col = divmod(i, images_per_row)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.show()

# Convert the images in the dataset dir into a tensor
#   Input: path to the dataset dir
#   Output: tensor of the images
def imagedir_to_tensor(path):
    return ImageFolder(path, transform=transforms.ToTensor())