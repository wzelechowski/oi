from torch.utils.data import DataLoader

from extracted_features_dataset import ExtractedFeaturesDataset
from plots import plot_decision_boundaries, plot_confusion_matrix, plot_feature_metrics, voronoi
from simple_mlp import SimpleMLP
from utlis import calculate_decision_boundaries, evaluate_model, train_model, calculate_feature_metrics


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
    pass