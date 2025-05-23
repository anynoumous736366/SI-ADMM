import numpy as np

from data import GetData
from method import Model
import argparse
import matplotlib.pyplot as plt
import os

class Experiments(object):

    def __init__(self, drug_drug_data, model_name='CTF', **kwargs):
        super().__init__()
        self.drug_drug_data = drug_drug_data
        self.model_name = model_name
        self.model = Model(model_name)
        self.parameters = kwargs


    def CV_triplet(self):
        k_folds = 5
        index_matrix = np.array(np.where(self.drug_drug_data.X == 1))
        positive_num = index_matrix.shape[1]
        sample_num_per_fold = int(positive_num / k_folds)

        np.random.seed(0)
        np.random.shuffle(index_matrix.T)

        all_metrics = []
        roc_curves = []
        pr_curves = []

        for k in range(k_folds):
            train_tensor = np.array(self.drug_drug_data.X, copy=True)
            if k != k_folds - 1:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_tensor[train_index] = 0
            S1 = np.mat(self.drug_drug_data.S1)
            S2 = np.mat(self.drug_drug_data.S2)

            predict_tensor = self.model()(train_tensor, S1, S2, **self.parameters)

            for i in range(10):
                *metrics, roc_curve, pr_curve = self.cv_tensor_model_evaluate(self.drug_drug_data.X, predict_tensor,
                                                                              train_index, i)
                all_metrics.append(metrics)
                roc_curves.append(roc_curve)
                pr_curves.append(pr_curve)

        all_metrics = np.array(all_metrics)
        mean_metrics = np.around(np.mean(all_metrics, axis=0), decimals=4)
        std_metrics = np.around(np.std(all_metrics, axis=0), decimals=4)

        return mean_metrics, std_metrics, roc_curves, pr_curves

    def cv_tensor_model_evaluate(self, association_tensor, predict_tensor, train_index, seed):
        test_po_num = np.array(train_index).shape[1]
        test_index = np.array(np.where(association_tensor == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        # print(np.where((negative_index-test_index)!=0))
        test_ne_index = tuple(test_index[:, :test_po_num])
        real_score = np.column_stack(
            (np.mat(association_tensor[test_ne_index].flatten()), np.mat(association_tensor[train_index].flatten())))
        predict_score = np.column_stack(
            (np.mat(predict_tensor[test_ne_index].flatten()), np.mat(predict_tensor[train_index].flatten())))
        # real_score and predict_score are array
        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        # Sort unique prediction scores for thresholding
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = np.mat(sorted_predict_score[(np.arange(1, 1000) * sorted_predict_score_num // 1000)])
        thresholds_num = thresholds.shape[1]

        # Generate binary predictions at all thresholds
        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        predict_score_matrix[predict_score_matrix < thresholds.T] = 0
        predict_score_matrix[predict_score_matrix >= thresholds.T] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        # ROC Coordinates
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T.A1
        y_ROC = ROC_dot_matrix[1].T.A1
        auc = 0.5 * np.sum((x_ROC[1:] - x_ROC[:-1]) * (y_ROC[1:] + y_ROC[:-1]))

        # PR Coordinates
        recall_list = tpr.A1
        precision_list = (TP / (TP + FP)).A1
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T.A1
        y_PR = PR_dot_matrix[1].T.A1
        aupr = 0.5 * np.sum((x_PR[1:] - x_PR[:-1]) * (y_PR[1:] + y_PR[:-1]))

        # Best F1 and other metrics
        f1_score_list = (2 * TP / (len(real_score.T) + TP - TN)).A1
        accuracy_list = ((TP + TN) / len(real_score.T)).A1
        specificity_list = (TN / (TN + FP)).A1
        max_index = np.argmax(f1_score_list)

        return aupr, auc, f1_score_list[max_index], accuracy_list[max_index], recall_list[max_index], specificity_list[
            max_index], precision_list[max_index], (x_ROC, y_ROC), (x_PR, y_PR)


def parse_arguments():
    parser = argparse.ArgumentParser(description="arguments for various versions of admm baselines")
    parser.add_argument("--data", type=str, default="x", help='x or y tensors')
    parser.add_argument("--model", type=str, nargs='+', default=["CTF"], help='model to be used')
    parser.add_argument("--rank", type=int, default=4, help='rank to be used')
    return parser.parse_args()


def save_results_2_txt(model_name, results_text, data_source, rank, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{model_name}_{data_source}_{rank}_results.txt")

    with open(result_path, "w") as f:
        f.write(results_text + "\n")

    print(f"Saved averaged metrics with std to {result_path}")


def average_curves_with_std(curves):
    x_common = np.linspace(0, 1, 100)
    y_all = []

    for x, y in curves:
        y_interp = np.interp(x_common, x, y)  # Interpolate to common x-axis
        y_all.append(y_interp)

    y_all = np.array(y_all)
    y_mean = np.mean(y_all, axis=0)
    y_std = np.std(y_all, axis=0)

    return x_common, y_mean, y_std


def save_curves_to_npz(model_name, data_source, roc_curves, pr_curves, rank, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name}_{data_source}_{rank}_curves.npz")

    # Average and compute std for ROC
    x_roc_avg, y_roc_avg, y_roc_std = average_curves_with_std(roc_curves)
    # Average and compute std for PR
    x_pr_avg, y_pr_avg, y_pr_std = average_curves_with_std(pr_curves)

    # Save all data
    np.savez_compressed(
        save_path,
        x_roc=x_roc_avg,
        y_roc=y_roc_avg,
        y_roc_std=y_roc_std,
        x_pr=x_pr_avg,
        y_pr=y_pr_avg,
        y_pr_std=y_pr_std
    )
    print(f"Saved averaged ROC and PR curves with std to {save_path}")


if __name__ == '__main__':
    args = parse_arguments()
    data_source = args.data
    rank = args.rank
    drug_drug_data = GetData(data_source)

    model = args.model

    params = {
        'CTF': dict(r=rank, mu=0.5, eta=0.2, alpha=0.5, beta=0.5, lam=0.5, xita=0.5, tol=1e-6, max_iter=200),
        'TFAI_CP_within_mod': dict(r=rank, alpha=0.5, beta=2.0, lam=0.001, tol=1e-6, max_iter=200),
        'TDRC': dict(r=rank, alpha=0.125, beta=1.0, lam=0.001, tol=1e-6, max_iter=200),
        'CP': dict(r=rank, tol=1e-6, max_iter=200)
    }

    for mod in model:
        if mod not in params:
            raise Exception('invalid model name')

        experiment = Experiments(drug_drug_data, model_name=mod, **params[mod])
        mean_results, std_results, roc_curves, pr_curves = experiment.CV_triplet()

        result_text = 'Mean:\n' + '\t'.join(map(str, mean_results)) + '\nSTD:\n' + '\t'.join(map(str, std_results))
        print(result_text)

        save_results_2_txt(mod, result_text, data_source, rank, output_dir='./baseline_output/')

        save_curves_to_npz(mod, data_source, roc_curves, pr_curves, rank, output_dir='./baseline_output/')

