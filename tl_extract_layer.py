import argparse
import pathlib
import torch
import numpy as np
import torch.nn as nn

import datasets
import models

from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import RepeatedStratifiedKFold
import torch.multiprocessing as mp


def main():
    # mp.set_start_method('fork', force=True)  # Prevent resource tracking issues
    args = parse_args()
    dir_extracted_layer = [
        'blocks.0.attn.proj',  
        'blocks.5.attn.proj',  
        'blocks.10.attn.proj',  
        'blocks.15.attn.proj',  
        'blocks.20.attn.proj',  
        'blocks.23.attn.proj',  
    ]

    datasets_path = pathlib.Path('/Volumes/T7Pawel/Datasets')
    dataset = datasets.get_dataset(args.dataset_name, datasets_path)

    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    fold_labels = [dataset.targets[i] for i in range(len(dataset))]
    device = torch.device(args.device)

    for layer_name in dir_extracted_layer:
        model = models.load_model(args.model_name)

        all_results = {
            "Transrate": []
        }

        for train_idx, test_idx in tqdm(kf.split(dataset, fold_labels)):
            test_dataset = Subset(dataset, test_idx)

            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

            model.eval()
            model.to(device)

            total_transrate = 0

            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    features = extract_features(args.model_name, model, inputs, layer_name)
                    centralized_normalized_features = centralize_and_normalize(features.cpu())
                    del inputs, features
                    transrate_value = transrate(centralized_normalized_features.cpu().numpy(), labels.cpu().numpy())
                    total_transrate += transrate_value
                    del centralized_normalized_features, transrate_value
                    torch.cuda.empty_cache() 

            all_results["Transrate"].append(total_transrate / len(test_dataloader))

        print(f"\n==== Cross-Validation Results: {layer_name} ====")
        print(f"Overall Average Transrate for {args.model_name}: {np.mean(all_results['Transrate']):.3f} ± {np.std(all_results['Transrate']):.3f}\n")
        print(all_results)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', choices=('APTOS2019', 'GLAUCOMA', 'JSIEC'), default='JSIEC')
    parser.add_argument('--model_name', choices=('ResNet50_clear', 'ResNet50', 'MAE_clear','MAE', 'RETFound_oct', 'RETFound_cfp'), default='RETFound_cfp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args


def extract_features(modelName, model, inputs, layerName):
    if modelName in ['RETFound_oct', 'RETFound_cfp']:
        return_layer = {layerName: 'extracted_flatten'}
    elif modelName in ['MAE', 'MAE_clear', 'ViT-Large']:
        return_layer = {'head': 'extracted_flatten'}
    elif modelName in ['ResNet50', 'ResNet50_clear']:
        return_layer = {'fc': 'extracted_flatten'}
    else:
        raise ValueError(f'There is no model defined as {modelName}')

    feature_extractor = create_feature_extractor(model, return_nodes=return_layer)
    features = feature_extractor(inputs)['extracted_flatten']
    return features


def centralize_and_normalize(features):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    if features.ndim > 2:
        features = features.flatten(start_dim=1)

    centralized_features = features - torch.mean(features, dim=0, keepdim=True)
    scale_factor = torch.linalg.norm(centralized_features, ord='fro') + 1e-6
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


def transrate_deep_layers(Z, y, eps=1E-4):
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


def coding_rate_deep_layers(Z, eps=1E-4):
    n, d = Z.shape
    Z_normalized = Z / np.sqrt(n * eps)

    try:
        U, S, Vt = np.linalg.svd(Z_normalized, full_matrices=False)
        logdet = np.sum(np.log(S + eps))
    except np.linalg.LinAlgError:
        print("Warning: SVD failed, falling back to standard determinant calculation")
        _, logdet = np.linalg.slogdet((1 / (n * eps)) * (Z.T @ Z) + eps * np.eye(d))

    return 0.5 * logdet


if __name__ == '__main__':
    main()
