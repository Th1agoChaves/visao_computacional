import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo no conjunto de teste e retorna loss e accuracy."""
    test_loss, test_accuracy= model.evaluate(X_test, y_test, verbose=1)
    print(f"Acurácia no teste: {test_accuracy * 100:.2f}%")
    print(f"Loss no teste: {test_loss:.4f}")
    return test_loss, test_accuracy

def plot_training_history(history, output_dir=None):
    """Plota acurácia e loss de treino/validação e salva figuras, se pedido."""
    figures = []
    plots = [
        ("accuracy", "val_accuracy", "Acurácia por época", "Acurácia", "accuracy.png"),
        ("loss", "val_loss", "Loss por época", "Loss", "loss.png"),
    ]

    for train_key, val_key, title, ylabel, filename in plots:
        figure, axis= plt.subplots(figsize=(8, 5))
        axis.plot(history.history[train_key], label=f"Treino: {ylabel}")
        axis.plot(history.history[val_key], label=f"Validação: {ylabel}")
        axis.set_title(title)
        axis.set_xlabel("Época")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()

        if output_dir is not None:
            figure.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        figures.append(figure)
        plt.close(figure)

    return figures