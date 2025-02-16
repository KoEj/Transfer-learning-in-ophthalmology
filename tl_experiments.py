import argparse
import pathlib
import torch
import numpy as np

import datasets
import models

from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def main():
    args = parse_args()

    datasets_path = pathlib.Path('/Datasets')
    dataset = datasets.get_dataset(args.dataset_name, datasets_path)

    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    fold_labels = [dataset.targets[i] for i in range(len(dataset))]

    device = torch.device(args.device)
    model = models.load_model(args.model_name)

    all_results = {
        "Transrate": [], 
        "H-Score": [], 
        "Classifier_Metrics": {
            "GNB": {"Accuracy": [], "Precision": [], "Recall": []},
            "KNN": {"Accuracy": [], "Precision": [], "Recall": []},
            "SVC": {"Accuracy": [], "Precision": [], "Recall": []}
        }
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, fold_labels)):
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        model.eval()
        model.to(device)

        train_features, train_labels, test_features, test_labels = [], [], [], []

        with torch.no_grad():
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                features = extract_features(args.model_name, model, inputs)
                train_features.append(features.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

        train_features = np.concatenate(train_features)
        train_labels = np.array(train_labels)
        pca = PCA(n_components=0.70, random_state=1234)
        train_features_pca = pca.fit_transform(train_features)

        classifiers = {
            "GNB": GaussianNB(),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVC": SVC(kernel='linear', random_state=1234)
        }

        for clf_name, clf in classifiers.items():
            clf.fit(train_features_pca, train_labels)

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                features = extract_features(args.model_name, model, inputs)
                test_features.append(features.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_features = np.concatenate(test_features)
        test_labels = np.array(test_labels)

        test_features_pca = pca.transform(test_features)

        for clf_name, clf in tqdm(classifiers.items(), desc=f"Classifier predictions fold {fold + 1}"):
            predictions = clf.predict(test_features_pca)
            accuracy = balanced_accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)

            all_results["Classifier_Metrics"][clf_name]["Accuracy"].append(accuracy)
            all_results["Classifier_Metrics"][clf_name]["Precision"].append(precision)
            all_results["Classifier_Metrics"][clf_name]["Recall"].append(recall)

        total_transrate = 0
        total_h_score = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                features = extract_features(args.model_name, model, inputs)
                centralized_normalized_features = centralize_and_normalize(features.cpu())

                transrate_value = transrate(centralized_normalized_features.cpu().numpy(), labels.cpu().numpy())
                h_score_value = h_score(features.cpu().numpy(), labels.cpu().numpy())

                total_transrate += transrate_value
                total_h_score += h_score_value

        all_results["Transrate"].append(total_transrate / len(test_dataloader))
        all_results["H-Score"].append(total_h_score / len(test_dataloader))

    print("\n==== Cross-Validation Results ====")
    print(f"Overall Average Transrate for {args.model_name}: {np.mean(all_results['Transrate']):.3f} ± {np.std(all_results['Transrate']):.3f}")
    print(f"Overall Average H-Score for {args.model_name}: {np.mean(all_results['H-Score']):.3f} ± {np.std(all_results['H-Score']):.3f}")
    print(all_results)

    for clf_name, metrics in all_results["Classifier_Metrics"].items():
        mean_acc = np.mean(metrics["Accuracy"])
        std_acc = np.std(metrics["Accuracy"])
        mean_precision = np.mean(metrics["Precision"])
        std_precision = np.std(metrics["Precision"])
        mean_recall = np.mean(metrics["Recall"])
        std_recall = np.std(metrics["Recall"])

        print(f"\n{clf_name} Classifier Metrics:")
        print(f" - Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f" - Precision: {mean_precision:.3f} ± {std_precision:.3f}")
        print(f" - Recall: {mean_recall:.3f} ± {std_recall:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', choices=('APTOS2019', 'GLAUCOMA', 'JSIEC'), default='JSIEC')
    parser.add_argument('--model_name', choices=('ResNet50_clear', 'ResNet50', 'MAE_clear','MAE', 'RETFound_oct', 'RETFound_cfp'), default='RETFound_cfp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args


def extract_features(modelName, model, inputs):
    if modelName == 'RETFound_oct' or modelName == 'RETFound_cfp':
        features = model.forward_features(inputs)
        return features
    elif modelName == 'MAE' or modelName == 'MAE_clear' or modelName == 'ViT-Large':
        return_layer = {'head': 'extracted_flatten'}
    elif modelName == 'ResNet50' or modelName == 'ResNet50_clear':
        return_layer = {'fc': 'extracted_flatten'}
    else:
        raise ValueError(f'There is no model defined as {modelName}')

    feature_extractor = create_feature_extractor(model, return_nodes=return_layer)
    features = feature_extractor(inputs)['extracted_flatten']
    return features


def centralize_and_normalize(features):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    centralized_features = features - torch.mean(features, dim=0, keepdim=True)
    scale_factor = torch.sqrt(torch.trace(centralized_features.T @ centralized_features))
    normalized_features = centralized_features / scale_factor
    return normalized_features


def transrate(Z, y, eps=1E-4):
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.
    K = int(y.max() + 1)

    for i in range(K):
        class_mask = (y == i)
        class_features = Z[class_mask, :]
        if len(class_features) > 0:
            RZY += coding_rate(class_features, eps)

    return RZ - RZY / K


def coding_rate(Z, eps=1E-4):
    n, d = Z.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
    return 0.5 * rate

def h_score(features: np.ndarray, labels: np.ndarray):
    f = features
    y = labels

    covf = np.cov(f, rowvar=False)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score


if __name__ == '__main__':
    main()
