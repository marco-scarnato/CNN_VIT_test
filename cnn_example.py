import kagglehub
import feature_engineering as fe
import tensorflow as tf
import cnn

# Download latest version
path = kagglehub.dataset_download("d4rklucif3r/cat-and-dogs")

print("Path to dataset files:", path)
path = path + "\\dataset\\"

train_dir, _, test_dir = fe.dataset_sub_division(path)

# Parametri del dataset
img_size = (256, 256)
batch_size = 8

# Caricamento automatico delle immagini da cartelle (senza PyTorch)
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Caricamento automatico delle immagini da cartelle (senza PyTorch)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Determina il numero di classi automaticamente
num_classes = len(train_ds.class_names)
print(f"Numero di classi: {num_classes}, Classi: {train_ds.class_names}")

# ✅ Convertire i target in one-hot encoding
def one_hot_encode(image, label):
    return image, tf.one_hot(label, depth=num_classes)

train_ds = train_ds.map(one_hot_encode)
test_ds = test_ds.map(one_hot_encode)

print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))

# Creazione del modello
model = cnn.CNN_model(num_classes).model

# Addestramento della rete neurale
model.fit(
    train_ds,
    epochs=10,  # Modifica il numero di epoche se necessario
)

test_loss, test_acc = model.evaluate(test_ds)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")