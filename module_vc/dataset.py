from tensorflow import keras

CLASS_NAMES= [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]

def load_cifar10():
    """Carrega CIFAR-10, normaliza as imagens e achata os rótulos.

    Returns:
        Tupla (X_train, y_train, X_test, y_test), com imagens float32
        no intervalo [0, 1] e rótulos inteiros unidimensionais.
    """
    (X_train, y_train), (X_test, y_test)= keras.datasets.cifar10.load_data()

    X_train=    X_train.astype("float32")/ 255.0
    X_test=     X_test.astype("float32")/ 255.0
    y_train=    y_train.flatten()
    y_test=     y_test.flatten()

    return X_train, y_train, X_test, y_test

def plot_sample_images(images, labels, output_path=None, count=9):
    """Exibe uma grade de imagens e, opcionalmente, salva a figura."""
    import matplotlib.pyplot as plt

    count= min(count, len(images))
    figure, axes= plt.subplots(3, 3, figsize=(10, 10))

    for index, axis in enumerate(axes.flat):
        if index>= count:
            axis.axis("off")
            continue
        axis.imshow(images[index])
        axis.set_title(CLASS_NAMES[int(labels[index])])
        axis.axis("off")

    figure.suptitle("Exemplos da base CIFAR-10", fontsize=16)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)