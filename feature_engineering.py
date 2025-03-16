import os
import matplotlib.pyplot as plt
import seaborn as sns

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