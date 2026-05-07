from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dbscan import DBScan
from kmeans import Kmeans
from SVM import SVM
from p1 import plots
from p1.KNN import KNN
from p1.MLP import MLP
from utils import load_and_preprocess_data, draw_plots, load_classification_data


def main():
    # load_random_sets()
    # load_reals_sets()
    run_experiments()

def load_random_sets():
    files = ["data_1/2_1.csv", "data_1/2_2.csv", "data_1/2_3.csv", "data_1/3_1.csv", "data_1/3_2.csv", "data_1/3_3.csv"]

    eps_range = np.round(np.arange(0.1, 1, 0.1), 2)

    for file in files:
        X_scaled, y_true = load_and_preprocess_data(file)

        kmeans_strategies = [Kmeans(n_clusters=k) for k in range(2, 10)]
        draw_plots(X_scaled, y_true, kmeans_strategies, file)

        dbscan_strategies = [DBScan(eps=eps, min_samples=1) for eps in eps_range]
        draw_plots(X_scaled, y_true, dbscan_strategies, file)

def load_reals_sets():
    eps_range = np.round(np.arange(0.1, 3, 0.3), 2)

    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data)
    y_iris = iris.target

    kmeans_iris = [Kmeans(n_clusters=k) for k in range(2, 10)]
    draw_plots(X_iris, y_iris, kmeans_iris, "Iris")

    dbscan_iris = [DBScan(eps=eps, min_samples=1) for eps in eps_range]
    draw_plots(X_iris, y_iris, dbscan_iris, "Iris")

    wine = load_wine()
    X_wine = StandardScaler().fit_transform(wine.data)
    y_wine = wine.target

    kmeans_wine = [Kmeans(n_clusters=k) for k in range(2, 10)]
    draw_plots(X_wine, y_wine, kmeans_wine, "Wine")

    dbscan_wine = [DBScan(eps=eps, min_samples=1) for eps in eps_range]
    draw_plots(X_wine, y_wine, dbscan_wine, "Wine")

def run_experiments():
    files = ["data_1/2_3.csv"]
    for file in files:
        # run_experiment_1(file)
        # run_experiment_2_knn(file)
        # run_experiment_2_svm(file)
        # run_experiment_2_mlp(file)
        # run_experiment_3_knn(file)
        # run_experiment_3_svm(file)
        # run_experiment_3_mlp(file)
        # run_experiment_4(file)
        run_real_data_experiments()

def run_experiment_1(filepath):
    X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=0.8)

    c_values = [1.0, 10.0, 100.0]
    for kernel in ["linear", "rbf"]:
        best_train_acc = -1
        best_strategy = None

        for c in c_values:
            strategy = SVM(kernel=kernel, C=c)
            strategy.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, strategy.predict(X_train))

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_strategy = strategy

        test_acc = accuracy_score(y_test, best_strategy.predict(X_test))
        title = (f"{best_strategy.get_algorithm_name()} - C={best_strategy.get_param_value()}\n"
                 f"Acc Trening: {best_train_acc:.2f} | Acc Test: {test_acc:.2f}")

        plots.classification_decision_boundaries(
            best_strategy.model, X_train, y_train, X_test, y_test, title
        )

    neurons_list = [2, 4, 8, 10]

    for activation in ['identity', 'relu']:
        best_train_acc = -1
        best_strategy = None

        for n in neurons_list:
            strategy = MLP(activation=activation, hidden_layer_sizes=(n,))
            strategy.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, strategy.predict(X_train))

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_strategy = strategy

        test_acc = accuracy_score(y_test, best_strategy.predict(X_test))
        title = (f"{best_strategy.get_algorithm_name()} - Neurony={best_strategy.get_param_value()}\n"
                 f"Acc Trening: {best_train_acc:.2f} | Acc Test: {test_acc:.2f}")

        plots.classification_decision_boundaries(
            best_strategy.model, X_train, y_train, X_test, y_test, title
        )


def run_experiment_2_knn(filepath):
    X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=0.8)

    neighbours_list = [1, 3, 5, 9, 15, 25, 50]
    train_accs = []
    test_accs = []
    models = []

    for n in neighbours_list:
        strategy = KNN(n_neighbors=n)
        strategy.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, strategy.predict(X_train))
        test_acc = accuracy_score(y_test, strategy.predict(X_test))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        models.append(strategy)

    plots.accuracy_curve(
        neighbours_list, train_accs, test_accs,
        param_name='n_neighbours',
        title=f'K-NN: Wpływ n_neighbors na Accuracy ({filepath})'
    )

    best_idx = int(np.argmax(test_accs))

    cases_to_plot = [
        (0, "[Najmniejsza wartość]"),
        (best_idx, "[Najlepsza wartość]"),
        (len(neighbours_list) - 1, "[Największa wartość]")
    ]

    for idx, desc in cases_to_plot:
        strategy = models[idx]

        title = (f"K-NN (n={strategy.get_param_value()}) {desc}\n"
                 f"Trening: {train_accs[idx]:.2f} | Test: {test_accs[idx]:.2f}")

        plots.classification_decision_boundaries(
            strategy.model, X_train, y_train, X_test, y_test, title
        )

        plots.plot_confusion_matrix(
            strategy.model, X_test, y_test, title
        )


def run_experiment_2_svm(filepath):
    X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=0.8)

    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    train_accs = []
    test_accs = []
    models = []

    for c in c_values:
        strategy = SVM(kernel='rbf', C=c)
        strategy.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, strategy.predict(X_train))
        test_acc = accuracy_score(y_test, strategy.predict(X_test))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        models.append(strategy)

    plots.accuracy_curve(
        c_values, train_accs, test_accs,
        param_name='C',
        title=f'SVM: Wpływ parametru C na Accuracy ({filepath})',
        log_scale=True
    )

    best_idx = int(np.argmax(test_accs))

    cases_to_plot = [
        (0, "[Najmniejsza wartość]"),
        (best_idx, "[Najlepsza wartość na teście]"),
        (len(c_values) - 1, "[Największa wartość]")
    ]

    for idx, desc in cases_to_plot:
        strategy = models[idx]

        title = (f"SVM (C={strategy.get_param_value()}) {desc}\n"
                 f"Trening: {train_accs[idx]:.2f} | Test: {test_accs[idx]:.2f}")

        plots.classification_decision_boundaries(
            strategy.model, X_train, y_train, X_test, y_test, title
        )

        plots.plot_confusion_matrix(
            strategy.model, X_test, y_test, title
        )


def run_experiment_2_mlp(filepath):
    X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=0.8)

    neurons_list = [2, 5, 10, 20, 50]
    train_accs = []
    test_accs = []
    models = []

    for n in neurons_list:
        strategy = MLP(activation='relu', hidden_layer_sizes=(n,))
        strategy.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, strategy.predict(X_train))
        test_acc = accuracy_score(y_test, strategy.predict(X_test))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        models.append(strategy)

    plots.accuracy_curve(
        neurons_list, train_accs, test_accs,
        param_name='Liczba neuronów',
        title=f'MLP: Wpływ liczby neuronów na Accuracy ({filepath})'
    )

    best_idx = int(np.argmax(test_accs))

    cases_to_plot = [
        (0, "[Najmniejsza wartość]"),
        (best_idx, "[Najlepsza wartość na teście]"),
        (len(neurons_list) - 1, "[Największa wartość]")
    ]

    for idx, desc in cases_to_plot:
        strategy = models[idx]

        title = (f"MLP (Neurony={strategy.get_param_value()}) {desc}\n"
                 f"Trening: {train_accs[idx]:.2f} | Test: {test_accs[idx]:.2f}")

        plots.classification_decision_boundaries(
            strategy.model, X_train, y_train, X_test, y_test, title
        )

        plots.plot_confusion_matrix(
            strategy.model, X_test, y_test, title
        )


def run_experiment_3_knn(filepath):
    X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=0.2, test_size=0.2)

    max_n = min(50, len(X_train))
    neighbours_list = [n for n in [1, 3, 5, 9, 15, 25, 50] if n <= max_n]

    train_accs = []
    test_accs = []
    models = []

    for n in neighbours_list:
        strategy = KNN(n_neighbors=n)
        strategy.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, strategy.predict(X_train))
        test_acc = accuracy_score(y_test, strategy.predict(X_test))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        models.append(strategy)

    plots.accuracy_curve(
        neighbours_list, train_accs, test_accs,
        param_name='n_neighbours',
        title=f'K-NN: Wpływ n_neighbors ({filepath})'
    )

    best_idx = int(np.argmax(test_accs))

    cases_to_plot = [
        (0, "[Najmniejsza wartość]"),
        (best_idx, "[Najlepsza wartość]"),
        (len(neighbours_list) - 1, "[Największa wartość]")
    ]

    for idx, desc in cases_to_plot:
        strategy = models[idx]
        title = (f"K-NN (n={strategy.get_param_value()}) {desc}\n"
                 f"Trening: {train_accs[idx]:.2f} | Test: {test_accs[idx]:.2f}")

        plots.classification_decision_boundaries(strategy.model, X_train, y_train, X_test, y_test, title)
        plots.plot_confusion_matrix(strategy.model, X_test, y_test, title)


def run_experiment_3_svm(filepath):
    X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=0.2, test_size=0.2)

    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    train_accs = []
    test_accs = []
    models = []

    for c in c_values:
        strategy = SVM(kernel='rbf', C=c)
        strategy.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, strategy.predict(X_train))
        test_acc = accuracy_score(y_test, strategy.predict(X_test))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        models.append(strategy)

    plots.accuracy_curve(
        c_values, train_accs, test_accs,
        param_name='C',
        title=f'SVM: Wpływ C ({filepath})',
        log_scale=True
    )

    best_idx = int(np.argmax(test_accs))

    cases_to_plot = [
        (0, "[Najmniejsza wartość]"),
        (best_idx, "[Najlepsza wartość]"),
        (len(c_values) - 1, "[Największa wartość]")
    ]

    for idx, desc in cases_to_plot:
        strategy = models[idx]
        title = (f"SVM (C={strategy.get_param_value()}) {desc}\n"
                 f"Trening: {train_accs[idx]:.2f} | Test: {test_accs[idx]:.2f}")

        plots.classification_decision_boundaries(strategy.model, X_train, y_train, X_test, y_test, title)
        plots.plot_confusion_matrix(strategy.model, X_test, y_test, title)


def run_experiment_3_mlp(filepath):
    X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=0.2, test_size=0.2)

    neurons_list = [2, 5, 10, 20, 50]
    train_accs = []
    test_accs = []
    models = []

    for n in neurons_list:
        strategy = MLP(activation='relu', hidden_layer_sizes=(n,))
        strategy.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, strategy.predict(X_train))
        test_acc = accuracy_score(y_test, strategy.predict(X_test))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        models.append(strategy)

    plots.accuracy_curve(
        neurons_list, train_accs, test_accs,
        param_name='Liczba neuronów',
        title=f'MLP: Wpływ neuronów ({filepath})'
    )

    best_idx = int(np.argmax(test_accs))

    cases_to_plot = [
        (0, "[Najmniejsza wartość]"),
        (best_idx, "[Najlepsza wartość]"),
        (len(neurons_list) - 1, "[Największa wartość]")
    ]

    for idx, desc in cases_to_plot:
        strategy = models[idx]
        title = (f"MLP (Neurony={strategy.get_param_value()}) {desc}\n"
                 f"Trening: {train_accs[idx]:.2f} | Test: {test_accs[idx]:.2f}")

        plots.classification_decision_boundaries(strategy.model, X_train, y_train, X_test, y_test, title)
        plots.plot_confusion_matrix(strategy.model, X_test, y_test, title)


def run_experiment_4(filepath, best_neurons_exp2=2, best_neurons_exp3=20, n_epochs=1000):
    train_sizes = [0.8, 0.2]

    for t_size in train_sizes:
        if t_size == 0.8:
            n_neurons = best_neurons_exp2
        else:
            n_neurons = best_neurons_exp3

        X_train, X_test, y_train, y_test = load_classification_data(filepath, train_size=t_size)
        classes = np.unique(y_train)

        mlp_strategy = MLP(hidden_layer_sizes=(n_neurons,), activation='relu', random_state=42)

        history = {'train': [], 'test': []}
        saved_models = {}
        best_test_acc = -1

        for epoch in range(1, n_epochs + 1):
            mlp_strategy.partial_fit(X_train, y_train, classes=classes)

            curr_train_acc = accuracy_score(y_train, mlp_strategy.predict(X_train))
            curr_test_acc = accuracy_score(y_test, mlp_strategy.predict(X_test))

            history['train'].append(curr_train_acc)
            history['test'].append(curr_test_acc)

            if epoch == 1:
                saved_models['pierwsza'] = (deepcopy(mlp_strategy), epoch, curr_train_acc, curr_test_acc)

            if curr_test_acc > best_test_acc:
                best_test_acc = curr_test_acc
                saved_models['najlepsza'] = (deepcopy(mlp_strategy), epoch, curr_train_acc, curr_test_acc)

            if epoch == n_epochs:
                saved_models['ostatnia'] = (deepcopy(mlp_strategy), epoch, curr_train_acc, curr_test_acc)

        epochs_list = list(range(1, n_epochs + 1))
        plots.learning_curve(epochs_list, history['train'], history['test'], f"Proces nauki MLP (train_size={t_size})")

        for key in ['pierwsza', 'najlepsza', 'ostatnia']:
            model_wrapper, ep, tr_acc, te_acc = saved_models[key]
            title = f"Exp 4: Epoka {key} ({ep})\nTrening: {tr_acc:.2f} | Test: {te_acc:.2f}"

            plots.classification_decision_boundaries(model_wrapper.model, X_train, y_train, X_test, y_test, title)
            plots.plot_confusion_matrix(model_wrapper.model, X_test, y_test, title)

        results_summary = []

        for i in range(10):
            run_mlp = MLP(hidden_layer_sizes=(n_neurons,), activation='relu', random_state=i)

            run_best_acc = -1
            run_best_epoch = 0
            run_data = {}

            for epoch in range(1, n_epochs + 1):
                run_mlp.partial_fit(X_train, y_train, classes=classes)

                te_acc = accuracy_score(y_test, run_mlp.predict(X_test))
                tr_acc = accuracy_score(y_train, run_mlp.predict(X_train))

                if epoch == 1:
                    run_data['first'] = (tr_acc, te_acc)

                if te_acc > run_best_acc:
                    run_best_acc = te_acc
                    run_best_epoch = epoch
                    run_data['best'] = (tr_acc, te_acc, run_best_epoch)

                if epoch == n_epochs:
                    run_data['last'] = (tr_acc, te_acc)

            results_summary.append({
                'Próba': i + 1,
                'Ep1_Tr': run_data['first'][0], 'Ep1_Te': run_data['first'][1],
                'Best_Tr': run_data['best'][0], 'Best_Te': run_data['best'][1], 'Best_Ep': run_data['best'][2],
                'Last_Tr': run_data['last'][0], 'Last_Te': run_data['last'][1]
            })

        df_results = pd.DataFrame(results_summary)
        print(df_results.to_string(index=False))


def run_real_data_experiments():
    print(f"\n{'=' * 50}\nEksperyment 5: Dane rzeczywiste (Wszystkie miary)\n{'=' * 50}")
    results = []

    datasets = [
        ("Iris", load_iris(return_X_y=True)),
        ("Wine", load_wine(return_X_y=True))
    ]

    for dataset_name, (X, y) in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        knn = KNN(n_neighbors=5).fit(X_train, y_train)

        svm = SVM(kernel='rbf', C=1.0).fit(X_train, y_train)

        mlp = MLP(hidden_layer_sizes=(50,), activation='relu').fit(X_train, y_train)

        models = {'K-NN': knn, 'SVM': svm, 'MLP': mlp}

        for model_name, model in models.items():
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            results.append({
                'Zbiór': dataset_name,
                'Model': model_name,
                'Accuracy': acc,
                'Precision (Macro)': prec,
                'Recall (Macro)': rec,
                'F1 Score (Macro)': f1
            })

    df = pd.DataFrame(results)

    print("\nSzczegółowe wyniki klasyfikacji na danych rzeczywistych:")
    print(df.to_string(index=False, float_format="{:.3f}".format))

    pivot_df = df.pivot(index='Zbiór', columns='Model', values='Accuracy')
    plots.real_data_bar_chart(pivot_df, title="Accuracy na zbiorach rzeczywistych")


if __name__ == '__main__':
    main()