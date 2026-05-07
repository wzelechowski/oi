import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score, \
    confusion_matrix
from torch import nn, optim

def train_model(model, train_loader, learning_rate=0.01, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoka [{epoch + 1}/{num_epochs}], Strata: {epoch_loss:.4f}, Dokładność: {epoch_acc:.2f}%")

    print("Trening zakończony!")
    return model


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

    print(f"\nWynik na zbiorze TESTOWYM: {accuracy:.2f}%")

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