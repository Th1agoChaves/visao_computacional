from pathlib import Path

from module_vc.dataset import load_cifar10, plot_sample_images
from module_vc.modeling.evaluate import evaluate_model, plot_training_history
from models.model import build_model
from module_vc.modeling.predict import plot_predictions
from module_vc.modeling.train import train_model

PROJECT_ROOT= Path(__file__).resolve().parents[2]
MODELS_DIR= PROJECT_ROOT/ "models"
FIGURES_DIR= PROJECT_ROOT/ "reports"/ "figures"

def main():
    """Carrega dados, treina, avalia, gera figuras e salva o modelo."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")
    print(f"Formato de X_test: {X_test.shape}")
    print(f"Formato de y_test: {y_test.shape}")

    plot_sample_images(X_train, y_train, FIGURES_DIR/ "cifar10_examples.png")

    model= build_model()
    model.summary()

    history= train_model(X_train=X_train, y_train=y_train, model=model)
    plot_training_history(history, FIGURES_DIR)
    evaluate_model(model, X_test, y_test)
    plot_predictions(model, X_test, y_test, FIGURES_DIR/ "cifar10_predictions.png")

    model_path= MODELS_DIR/ "cifar10_cnn.keras"
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")

if __name__ == "__main__":
    main()