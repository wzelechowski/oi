import numpy as np
import torch
import torchvision
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from plots import plot_confusion_matrix, plot_decision_boundaries, plot_learning_curves

mnist_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_aug1 = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

imagenette_base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

imagenette_aug1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def train_model(model, train_loader, test_loader, learning_rate=0.001, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_acc_history = []
    test_acc_history = []
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        acc_tr, _ = evaluate_model(model, train_loader)
        acc_te, _ = evaluate_model(model, test_loader)
        train_acc_history.append(acc_tr)
        test_acc_history.append(acc_te)
        print(f"Epoka [{epoch + 1}/{num_epochs}], Acc Train: {acc_tr:.2f}%, Acc Test: {acc_te:.2f}%")
    return model, train_acc_history, test_acc_history

def calculate_feature_metrics(X_dict, y_true):
    print(f"\n{'Zbiór cech':<15} | {'Silhouette':<10} | {'ARI':<10} | {'Homogeneity':<12} | {'Completeness':<12}")
    print("-" * 65)
    results = {}
    for name, X in X_dict.items():
        sil_score = silhouette_score(X, y_true)
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(X)
        ari = adjusted_rand_score(y_true, predicted_labels)
        hom = homogeneity_score(y_true, predicted_labels)
        comp = completeness_score(y_true, predicted_labels)
        print(f"{name:<15} | {sil_score:>10.4f} | {ari:>10.4f} | {hom:>12.4f} | {comp:>12.4f}")
        results[name] = {
            'Silhouette': sil_score,
            'ARI': ari,
            'Homogeneity': hom,
            'Completeness': comp
        }
    return results

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100 * correct / len(all_labels)
    # print(f"\nWynik na zbiorze TESTOWYM: {accuracy:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm

def calculate_decision_boundaries(model, X, h=0.05):
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(grid_tensor)
        _, preds = torch.max(outputs, 1)
    Z = preds.numpy().reshape(xx.shape)
    return xx, yy, Z

def get_balanced_subset(dataset, total_samples, num_classes=10):
    if total_samples == 'all':
        return dataset
    samples_per_class = total_samples // num_classes
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, '_labels'):
        targets = np.array(dataset._labels)
    else:
        targets = np.array([label for _, label in dataset])
    indices = []
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        chosen_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(chosen_indices)
    return Subset(dataset, indices)

def get_mnist_datasets(train_transform=mnist_base_transform):
    print("Pobieranie zbioru MNIST...")
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_base_transform)
    return train_ds, test_ds

def get_imagenette_datasets(train_transform=imagenette_base_transform):
    print("Pobieranie zbioru Imagenette...")
    try:
        train_ds = torchvision.datasets.Imagenette(root='./data', split='train', size='160px', download=True, transform=train_transform)
        test_ds = torchvision.datasets.Imagenette(root='./data', split='val', size='160px', download=True, transform=imagenette_base_transform)
        return train_ds, test_ds
    except AttributeError:
        return None, None

def run_experiment(name, model, train_ds, test_ds, is_2d, epochs=10, batch_size=64):
    print(f"\n" + "=" * 50)
    print(f"URUCHAMIAM: {name}")
    print("=" * 50)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    print(f"Rozpoczynam trening ({epochs} epok)...")
    trained_model, train_acc_hist, test_acc_hist = train_model(
        model, train_loader, test_loader, learning_rate=0.001, num_epochs=epochs
    )
    print("Generowanie krzywych uczenia...")
    plot_learning_curves(train_acc_hist, test_acc_hist, title=f"Krzywe uczenia - {name}")
    print("Generowanie macierzy (TRENING)...")
    acc_tr, cm_tr = evaluate_model(trained_model, train_loader)
    plot_confusion_matrix(cm_tr, title=f"Macierz Pomyłek (TRENING) - {name}\n(Acc: {acc_tr:.2f}%)")
    print("Generowanie macierzy (TEST)...")
    acc_te, cm_te = evaluate_model(trained_model, test_loader)
    plot_confusion_matrix(cm_te, title=f"Macierz Pomyłek (TEST) - {name}\n(Acc: {acc_te:.2f}%)")
    if is_2d:
        print(f"Generowanie granic decyzyjnych dla {name}...")
        trained_model.eval()
        features_list, labels_list = [], []
        plot_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            for images, labels in plot_loader:
                _, features = trained_model(images, return_features=True)
                features_list.append(features)
                labels_list.append(labels)
                if len(features_list) * batch_size >= 1000:
                    break
        X_plot = torch.cat(features_list).numpy()
        y_plot = torch.cat(labels_list).numpy()
        class ClassifierWrapper(torch.nn.Module):
            def __init__(self, classifier_layer):
                super().__init__()
                self.layer = classifier_layer
            def forward(self, x):
                return self.layer(x)
        clf_wrapper = ClassifierWrapper(trained_model.final_classifier)
        xx, yy, Z = calculate_decision_boundaries(clf_wrapper, X_plot)
        if "MNIST" in name:
            labels_names = [f"Cyfra {i}" for i in range(10)]
        else:
            labels_names = ['Ryba (Tench)', 'Pies (Springer)', 'Odtwarzacz Kaset', 'Piła łańcuchowa',
                            'Kościół', 'Wolsztyn', 'Śmieciarka', 'Dystrybutor paliwa', 'Piłka golfowa', 'Spadochron']
        plot_decision_boundaries(xx, yy, Z, X_plot, y_plot, title=f"Granice decyzyjne - {name}",
                                 class_names=labels_names)