import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import os

def extract_784_features(img_tensor):
    return img_tensor.view(-1)


def extract_2_features(img_tensor):
    img = img_tensor.squeeze()
    ink_amount = img.mean().item()

    left_half = img[:, :14]
    right_half = img[:, 14:]
    right_half_flipped = torch.flip(right_half, dims=[1])
    symmetry = torch.abs(left_half - right_half_flipped).mean().item()

    return torch.tensor([ink_amount, symmetry], dtype=torch.float32)


def extract_16_features(img_tensor):
    img = img_tensor.squeeze()
    features = []
    for i in range(4):
        for j in range(4):
            block = img[i * 7:(i + 1) * 7, j * 7:(j + 1) * 7]
            features.append(block.mean().item())
    return torch.tensor(features, dtype=torch.float32)



def process_and_save_mnist(save_dir='./processed_data'):
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def get_raw_features(dataset):
        X_784, X_2, X_16, Y = [], [], [], []
        for i in range(len(dataset)):
            img, label = dataset[i]
            X_784.append(extract_784_features(img))
            X_2.append(extract_2_features(img))
            X_16.append(extract_16_features(img))
            Y.append(label)
            if (i + 1) % 10000 == 0:
                print(f"  Przetworzono {i + 1} / {len(dataset)}")
        return torch.stack(X_784), torch.stack(X_2), torch.stack(X_16), torch.tensor(Y, dtype=torch.long)

    print("Ekstrakcja cech ze zbioru TRENINGOWEGO...")
    tr_784_raw, tr_2_raw, tr_16_raw, y_train = get_raw_features(train_data)

    print("Ekstrakcja cech ze zbioru TESTOWEGO...")
    te_784_raw, te_2_raw, te_16_raw, y_test = get_raw_features(test_data)

    print("Skalowanie cech...")

    scaler_2 = StandardScaler()
    scaler_16 = StandardScaler()
    scaler_784 = StandardScaler()

    x_train_2 = scaler_2.fit_transform(tr_2_raw.numpy())
    x_test_2 = scaler_2.transform(te_2_raw.numpy())

    x_train_16 = scaler_16.fit_transform(tr_16_raw.numpy())
    x_test_16 = scaler_16.transform(te_16_raw.numpy())

    x_train_784 = scaler_784.fit_transform(tr_784_raw.numpy())
    x_test_784 = scaler_784.transform(te_784_raw.numpy())

    print("Zapisywanie plików .pt...")
    torch.save(y_train, os.path.join(save_dir, 'y_train.pt'))
    torch.save(y_test, os.path.join(save_dir, 'y_test.pt'))

    torch.save(torch.tensor(x_train_2, dtype=torch.float32), os.path.join(save_dir, 'x_train_2.pt'))
    torch.save(torch.tensor(x_test_2, dtype=torch.float32), os.path.join(save_dir, 'x_test_2.pt'))

    torch.save(torch.tensor(x_train_16, dtype=torch.float32), os.path.join(save_dir, 'x_train_16.pt'))
    torch.save(torch.tensor(x_test_16, dtype=torch.float32), os.path.join(save_dir, 'x_test_16.pt'))

    torch.save(torch.tensor(x_train_784, dtype=torch.float32), os.path.join(save_dir, 'x_train_784.pt'))
    torch.save(torch.tensor(x_test_784, dtype=torch.float32), os.path.join(save_dir, 'x_test_784.pt'))

    print("MNIST gotowy i przeskalowany!\n")


def prepare_iris_wine(save_dir='./processed_data'):
    os.makedirs(save_dir, exist_ok=True)

    datasets = {
        'iris': load_iris(),
        'wine': load_wine()
    }

    for name, data in datasets.items():
        print(f"Przetwarzanie i skalowanie zbioru: {name}")
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        torch.save(torch.tensor(X_train_scaled, dtype=torch.float32), f'{save_dir}/x_{name}_train.pt')
        torch.save(torch.tensor(X_test_scaled, dtype=torch.float32), f'{save_dir}/x_{name}_test.pt')
        torch.save(torch.tensor(y_train, dtype=torch.long), f'{save_dir}/y_{name}_train.pt')
        torch.save(torch.tensor(y_test, dtype=torch.long), f'{save_dir}/y_{name}_test.pt')

    print("Zbiory Iris i Wine gotowe i przeskalowane!")