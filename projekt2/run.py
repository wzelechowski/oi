import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from extracted_features_dataset import ExtractedFeaturesDataset
from imagenette_2d_cnn import Imagenette2DCNN
from imagenette_standard_cnn import ImagenetteStandardCNN
from mnist2dcnn import Mnist2DCNN
from mnist_standard_cnn import MnistStandardCNN
from plots import plot_decision_boundaries, plot_confusion_matrix, plot_feature_metrics, voronoi, \
    plot_augmentation_examples
from simple_mlp import SimpleMLP
from utlis import (
    get_mnist_datasets, get_imagenette_datasets, get_balanced_subset,
    train_model, evaluate_model,
    mnist_base_transform, mnist_aug1,
    imagenette_base_transform, imagenette_aug1, run_experiment, calculate_decision_boundaries,
    calculate_feature_metrics
)


def run1():
    # process_and_save_mnist()
    # prepare_iris_wine()

    print("Wczytywanie zbiorów danych do analizy separowalności...")
    ds_2 = ExtractedFeaturesDataset('./processed_data/x_train_2.pt', './processed_data/y_train.pt')
    ds_16 = ExtractedFeaturesDataset('./processed_data/x_train_16.pt', './processed_data/y_train.pt')
    ds_784 = ExtractedFeaturesDataset('./processed_data/x_train_784.pt', './processed_data/y_train.pt')

    subset_size = 2000
    X_dict = {
        '2 cechy': ds_2.X[:subset_size].numpy(),
        '16 cech': ds_16.X[:subset_size].numpy(),
        '784 cech': ds_784.X[:subset_size].numpy()
    }
    y_true = ds_2.y[:subset_size].numpy()

    results = calculate_feature_metrics(X_dict, y_true)
    plot_feature_metrics(results)
    voronoi(ds_2.X[:1000].numpy(), ds_2.y[:1000].numpy(), title="MNIST: 2 cechy")

    configs = [
        ("MNIST 2 Cechy", "x_train_2.pt", "x_test_2.pt", "y_train.pt", "y_test.pt", 2, 10),
        ("MNIST 16 Cech", "x_train_16.pt", "x_test_16.pt", "y_train.pt", "y_test.pt", 16, 10),
        ("MNIST 784 Cechy", "x_train_784.pt", "x_test_784.pt", "y_train.pt", "y_test.pt", 784, 10),
        ("Iris", "x_iris_train.pt", "x_iris_test.pt", "y_iris_train.pt", "y_iris_test.pt", 4, 3),
        ("Wine", "x_wine_train.pt", "x_wine_test.pt", "y_wine_train.pt", "y_wine_test.pt", 13, 3)
    ]

    for name, x_tr, x_te, y_tr, y_te, in_size, num_cls in configs:
        print(f"\n" + "=" * 50)
        print(f"URUCHAMIAM: {name}")
        print("=" * 50)

        train_ds = ExtractedFeaturesDataset(f'./processed_data/{x_tr}', f'./processed_data/{y_tr}')
        test_ds = ExtractedFeaturesDataset(f'./processed_data/{x_te}', f'./processed_data/{y_te}')

        b_size = 16 if in_size < 20 else 64
        train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=b_size, shuffle=False)

        hidden = 64 if in_size > 100 else 16
        model = SimpleMLP(input_size=in_size, hidden_size=hidden, num_classes=num_cls)

        if in_size == 784:
            epochs = 10
        elif in_size < 20 and num_cls == 3:
            epochs = 20
        else:
            epochs = 30

        trained_model = train_model(model, train_loader, learning_rate=0.01, num_epochs=epochs)

        acc_tr, cm_tr = evaluate_model(trained_model, train_loader)
        plot_confusion_matrix(cm_tr, title=f"Macierz Pomyłek (TRENING) - {name} (Acc: {acc_tr:.2f}%)")

        acc_te, cm_te = evaluate_model(trained_model, test_loader)
        plot_confusion_matrix(cm_te, title=f"Macierz Pomyłek (TEST) - {name} (Acc: {acc_te:.2f}%)")

        if in_size == 2 and name == "MNIST 2 Cechy":
            print("Generowanie granic decyzyjnych dla 2 cech...")
            X_test_plot = test_ds.X.numpy()
            y_test_plot = test_ds.y.numpy()

            xx, yy, Z = calculate_decision_boundaries(trained_model, X_test_plot)

            plot_decision_boundaries(xx, yy, Z, X_test_plot, y_test_plot, title="Granice decyzyjne MLP (MNIST 2 cechy)")


def run2():
    mnist_train, mnist_test = get_mnist_datasets()
    imagenette_train, imagenette_test = get_imagenette_datasets()

    run_experiment("MNIST - Standard CNN", MnistStandardCNN(), mnist_train, mnist_test, is_2d=False, epochs=10)
    run_experiment("MNIST - CNN 2 Cechy", Mnist2DCNN(), mnist_train, mnist_test, is_2d=True, epochs=15)

    if imagenette_train is not None:
        run_experiment("Imagenette - Standard CNN", ImagenetteStandardCNN(), imagenette_train, imagenette_test,
                       is_2d=False, epochs=10)
        run_experiment("Imagenette - CNN 2 Cechy", Imagenette2DCNN(), imagenette_train, imagenette_test, is_2d=True,
                       epochs=15)


def run3():
    print("\n" + "="*50)
    print("ROZPOCZYNAM EKSPERYMENT 2: Augmentacja i rozmiar zbioru")
    print("="*50)
    raw_mnist = datasets.MNIST(root='./data', train=True, download=True)
    plot_augmentation_examples(raw_mnist, mnist_aug1, title="MNIST Augmentacja", is_mnist=True)
    raw_imagenette = datasets.Imagenette(root='./data', split='train', size='160px', download=True)
    plot_augmentation_examples(raw_imagenette, imagenette_aug1, title="Imagenette Augmentacja", is_mnist=False)
    experiments = [
        ("MNIST Standard CNN", lambda: MnistStandardCNN(), get_mnist_datasets,
         {"Brak": mnist_base_transform, "Augmentacja": mnist_aug1}, 60000, 10),

        ("MNIST 2D CNN", lambda: Mnist2DCNN(), get_mnist_datasets,
         {"Brak": mnist_base_transform, "Augmentacja": mnist_aug1}, 60000, 15),

        ("Imagenette Standard CNN", lambda: ImagenetteStandardCNN(), get_imagenette_datasets,
         {"Brak": imagenette_base_transform, "Augmentacja": imagenette_aug1}, 9469, 10),

        ("Imagenette 2D CNN", lambda: Imagenette2DCNN(), get_imagenette_datasets,
         {"Brak": imagenette_base_transform, "Augmentacja": imagenette_aug1}, 9469, 15)
    ]
    subset_sizes = ['all', 100, 200, 1000]
    num_repetitions = 10
    for exp_name, model_factory, get_data_fn, transforms_dict, total_size, base_epochs in experiments:
        print(f"\n\n>>> BADA: {exp_name} <<<")
        print(f"{'Augmentacja':<15} | {'all':<15} | {'100':<15} | {'200':<15} | {'1000':<15}")
        print("-" * 85)
        for aug_name, transform in transforms_dict.items():
            results_row = [aug_name.ljust(15)]
            train_full_ds, test_ds = get_data_fn(train_transform=transform)
            if train_full_ds is None:
                continue
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
            for size in subset_sizes:
                if size == 'all':
                    epochs = base_epochs
                else:
                    epochs = min(base_epochs * (total_size // size), 100)
                accuracies = []
                best_acc = 0.0
                for i in range(num_repetitions):
                    model = model_factory()
                    subset = get_balanced_subset(train_full_ds, size)
                    train_loader = DataLoader(subset, batch_size=32, shuffle=True)
                    trained_model, _, _ = train_model(model, train_loader, test_loader, num_epochs=epochs)
                    acc_te, _ = evaluate_model(trained_model, test_loader)
                    accuracies.append(acc_te)
                    if acc_te > best_acc:
                        best_acc = acc_te
                        safe_name = f"{exp_name.replace(' ', '_')}_{aug_name}_{size}.pth"
                        torch.save(trained_model.state_dict(), safe_name)
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                results_row.append(f"{mean_acc:.2f}±{std_acc:.2f}%".ljust(15))
            print(" | ".join(results_row))