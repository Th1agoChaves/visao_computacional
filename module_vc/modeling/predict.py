import numpy as np
import matplotlib.pyplot as plt

from module_vc.dataset import CLASS_NAMES

def predict_classes(model, images):
    """Retorna probabilidades e classes previstas para um lote de imagens."""
    probabilities= model.predict(images, verbose=0)
    predicted_labels= np.argmax(probabilities, axis=1)
    return probabilities, predicted_labels

def plot_predictions(model, images, true_labels, output_path=None, count=9):
    """Exibe as classes reais e previstas para imagens do conjunto fornecido."""
    count= min(count, len(images))
    _, predicted_labels= predict_classes(model, images[:count])
    figure, axes= plt.subplots(3, 3, figsize=(10, 10))

    for index, axis in enumerate(axes.flat):
        if index>= count:
            axis.axis("off")
            continue
        axis.imshow(images[index])
        real= CLASS_NAMES[int(true_labels[index])]
        predicted= CLASS_NAMES[int(predicted_labels[index])]
        axis.set_title(f"Real: {real}\nPredição: {predicted}")
        axis.axis("off")

    figure.suptitle("Previsões em imagens do conjunto de teste")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return predicted_labels