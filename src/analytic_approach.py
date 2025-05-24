import numpy as np
import pandas as pd
import os
import pickle
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def result_to_csv(real_x, real_y, pred_x, pred_y, x_test_indices, y_test_indices, full_save_dir, rnd_seed, rank, initial=False):
    # Extract values at the indices of interest from the original and modified tensors
    x_i, x_j, x_k = x_test_indices[:, 0], x_test_indices[:, 1], x_test_indices[:, 2]
    real_x_values = real_x[x_i, x_j, x_k]

    y_i, y_j, y_k = y_test_indices[:, 0], y_test_indices[:, 1], y_test_indices[:, 2]
    real_y_values = real_y[y_i, y_j, y_k]

    pred_x_values = pred_x[x_i, x_j, x_k]
    pred_y_values = pred_y[y_i, y_j, y_k]

    # Create a dataframe to store the indices and values
    data_x = {
        "Indices": [tuple(idx.tolist()) for idx in x_test_indices],
        "label": real_x_values.tolist(),
        "prediction": pred_x_values.tolist()
    }
    df_x = pd.DataFrame(data_x)

    data_y = {
        "Indices": [tuple(idx.tolist()) for idx in y_test_indices],
        "label": real_y_values.tolist(),
        "prediction": pred_y_values.tolist()
    }
    df_y = pd.DataFrame(data_y)

    # Save the dataframe to a CSV file
    if initial:
        csv_file_path_x = f"{full_save_dir}_test_x_result_{str(rnd_seed)}_rank_{rank}_initial.csv"
        csv_file_path_y = f"{full_save_dir}_test_y_result_{str(rnd_seed)}_rank_{rank}_initial.csv"
    else:
        csv_file_path_x = f"{full_save_dir}_test_x_result_{str(rnd_seed)}_rank_{rank}.csv"
        csv_file_path_y = f"{full_save_dir}_test_y_result_{str(rnd_seed)}_rank_{rank}.csv"
    df_x.to_csv(csv_file_path_x, index=False)
    df_y.to_csv(csv_file_path_y, index=False)
    print(f'result saved in {csv_file_path_x} and {csv_file_path_y}')

    return df_x, df_y


def load_si(args):
    si = args.si
    x_name = args.x_data
    y_name = args.y_data
    base_dir = args.base_dir
    Sa = []
    for i in si:
        df = pd.read_csv(base_dir + f'{x_name}_{y_name}_si_{i}.csv', index_col=0)
        similarity_matrix = df.values
        if not np.allclose(similarity_matrix, similarity_matrix.T):
            raise ValueError(f"si {i} not symmetric")
        Sa.append(similarity_matrix)
    return Sa


def load_tensor_x_y(args):
    base_dir = args.base_dir
    x_name = args.x_data
    y_name = args.y_data
    non_binary = args.nb

    if non_binary:
        with open(base_dir + f'{x_name}_non_binary_{y_name}_x.pickle', 'rb') as t_x:
            x = pickle.load(t_x)
    else:
        with open(base_dir + f'{x_name}_{y_name}_x.pickle', 'rb') as t_x:
            x = pickle.load(t_x)

    with open(base_dir + f'{x_name}_{y_name}_y.pickle', 'rb') as t_y:
        y = pickle.load(t_y)

    x_keys = sorted(x.keys(), key=lambda x: int(x.split()[0]))
    tensor_slices_x = [x[key].values for key in x_keys]
    for matrix in tensor_slices_x:
        if not np.allclose(matrix, matrix.T):
            raise ValueError("X matrix not symmetric")
    real_tensor_x = np.stack(tensor_slices_x, axis=-1)

    y_keys = sorted(y.keys(), key=lambda y: int(y.split()[0]))
    tensor_slices_y = [y[key].values for key in y_keys]
    for matrix in tensor_slices_y:
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Y matrix not symmetric")
    real_tensor_y = np.stack(tensor_slices_y, axis=-1)

    print(f"tensor X shape: {real_tensor_x.shape}")
    print(f"tensor Y shape: {real_tensor_y.shape}")

    return real_tensor_x, real_tensor_y


def f(U, D, V, W, Ci, Ui, Qi):
    first_term = np.linalg.norm(real_tensor_x - resemble_tensor(U, D, V)) \
                 + np.linalg.norm(real_tensor_y - resemble_tensor(U, D, W))
    second_term = 0
    for i in range(num_si):
        second_term += si_weight[i] * (np.linalg.norm(Sa[i] - Ci[i] @ Ui[i].T, ord='fro')
                                       + np.linalg.norm(Ui[i] @ Qi[i] - U, ord='fro'))
    ans = first_term + second_term
    return ans


def lagrangian(U, D, V, W, Ci, Ui, Qi, F, Y, UD_penalty, SI_penalty):
    first_term = f(U, D, V, W, Ci, Ui, Qi)
    second_term = np.sum(F * (D - U)) + 0.5 * UD_penalty * np.linalg.norm(D - U, ord='fro')
    third_term = 0
    for i in range(num_si):
        third_term += np.sum(Y[i] * (Ci[i] - Ui[i])) + 0.5 * SI_penalty * np.linalg.norm(Ci[i] - Ui[i], ord='fro')

    ans = first_term + second_term + third_term
    return first_term, ans


def create_scale_matrix(matrix):
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    col_sums = np.sum(matrix, axis=0)  # Sum over rows (columns-wise)
    scale_matrix = np.diag(col_sums)
    return scale_matrix


def khatri_rao(A, B):
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns")

    I, K = A.shape
    J = B.shape[0]

    result = np.zeros((I * J, K), dtype=A.dtype)
    for k in range(K):
        result[:, k] = np.kron(A[:, k], B[:, k])

    return result


def resemble_tensor(A, B, C):
    I, R = A.shape
    J = B.shape[0]
    K = C.shape[0]

    X_hat = np.zeros((I, J, K), dtype=A.dtype)

    for r in range(R):
        # Outer product of A[:, r], B[:, r], C[:, r]
        rank_one_tensor = np.outer(A[:, r], B[:, r])[:, :, None] * C[:, r][None, None, :]
        X_hat += rank_one_tensor

    return X_hat


def mode_n_matricization(X, mode):
    if X.ndim != 3:
        raise ValueError("Only 3D tensors are supported")
    if mode not in [1, 2, 3]:
        raise ValueError("mode must be 1, 2, or 3")

    I, J, K = X.shape

    if mode == 1:
        return X.transpose(0, 2, 1).reshape(I, J * K)
    elif mode == 2:
        return X.transpose(1, 2, 0).reshape(J, I * K)
    elif mode == 3:
        # Stack frontal slices, each flattened column-wise
        return np.stack([X[:, :, k].reshape(-1, order='F') for k in range(K)], axis=0)


def parse_arguments():
    parser = argparse.ArgumentParser(description="arguments for various versions of admm and its baseline")

    # arguments with default values
    parser.add_argument("--rank", type=int, default=3, help="rank number")
    parser.add_argument("--test_rnd_seed", type=int, default=111, help="rnd seed for test indeces")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="percentage of 1 data for test")
    parser.add_argument('--rnd_seeds', type=int, nargs='+', default=[123],
                        help="random seeds to use for different runs")
    parser.add_argument("--missing_rate", type=float, default=0.0,
                        help="missing rate to test robustness")
    parser.add_argument("--full_tensor", action="store_true", help="whether or not to use the full tensor (no test indices)")
    parser.add_argument("--base_dir", type=str, default="../data/", help='directory for loading data')
    parser.add_argument("--save_dir", type=str, default="../output/", help='directory for saving results')

    parser.add_argument("--train_iter", type=int, default=1000, help="how many training iteration")
    parser.add_argument("--tol", type=int, default=0.00000001, help="tolerance for ealry stopping")
    parser.add_argument("--tolerance", type=int, default=10, help="tolerance for ealry stopping")
    parser.add_argument("--drug_dir", type=str, default="../data/{}_{}_all_drugs.npy",
                        help='A frame string with {} placeholders for directory')
    parser.add_argument("--UDpenalty", type=float, default=0.0000001, help="penalty parameter for U = D")
    parser.add_argument("--SIpenalty", type=float, default=0.0000001, help="penalty parameter for Ui = Ci")
    parser.add_argument("--penalty_multiplier", type=float, default=1.15, help="mutiliper that makes the multipliers larger each iteration")
    parser.add_argument("--max_penalty", type=int, default=1000000000000, help="max penalty")
    parser.add_argument("--lr", type=float, default=0.005, help="tolerance to stop early")
    parser.add_argument('--si', type=int, nargs='+', default=[0, 1, 2, 3],
                        help="side informations to use")
    parser.add_argument('--si_weight', type=float, nargs='+', default=[0.1, 0.1, 0.1, 0.1],
                        help="weight for each side info")
    parser.add_argument("--x_weight", type=float, default=1.0, help="weight for X construction loss")
    parser.add_argument("--y_weight", type=float, default=1.0, help="weight for Y construction loss")

    # arguments that need to be filled in
    parser.add_argument("--x_data", type=str, required=True, help="data source for x")
    parser.add_argument("--y_data", type=str, required=True, help="data source for y")
    parser.add_argument("--nb", action="store_true", help="whether or not to use non-binary data for x")
    parser.add_argument("--save_name", type=str, required=True, help="base name for the algorithm")
    return parser.parse_args()


def generate_test_tensor(tensor, args):
    test_rnd_seed = args.test_rnd_seed
    test_ratio = args.test_ratio
    missing_rate = args.missing_rate
    assert tensor.ndim == 3
    rng = np.random.default_rng(test_rnd_seed)
    modified_tensor = tensor.copy()

    # Get indices of 1s and 0s
    one_indices = np.argwhere(tensor >= 1)
    zero_indices = np.argwhere(tensor == 0)

    # Keep only upper triangle to avoid symmetric duplicates
    one_indices = one_indices[one_indices[:, 0] <= one_indices[:, 1]]
    zero_indices = zero_indices[zero_indices[:, 0] <= zero_indices[:, 1]]

    # Select test samples
    num_ones_to_replace = int(test_ratio * len(one_indices))
    num_zeros_to_replace = num_ones_to_replace

    randomized_one_indices = rng.permutation(one_indices)
    randomized_zero_indices = rng.permutation(zero_indices)

    selected_one_indices = randomized_one_indices[:num_ones_to_replace]
    selected_zero_indices = randomized_zero_indices[:num_zeros_to_replace]

    # Zero out selected indices and symmetric counterparts
    if not args.full_tensor:
        for i, j, k in selected_one_indices:
            modified_tensor[i, j, k] = 0
            modified_tensor[j, i, k] = 0

        for i, j, k in selected_zero_indices:
            modified_tensor[i, j, k] = 0
            modified_tensor[j, i, k] = 0

    # Apply additional masking (simulate missing entries)
    if missing_rate > 0.0:
        remaining_ones = randomized_one_indices[num_ones_to_replace:]
        num_ones_to_mask = int(missing_rate * len(remaining_ones))
        masked_indices = remaining_ones[:num_ones_to_mask]

        for i, j, k in masked_indices:
            modified_tensor[i, j, k] = 0
            modified_tensor[j, i, k] = 0

    # Collect all test indices and their symmetric counterparts
    symmetric_one_indices = selected_one_indices[:, [1, 0, 2]]
    symmetric_zero_indices = selected_zero_indices[:, [1, 0, 2]]

    all_test_indices = np.concatenate([
        selected_one_indices,
        selected_zero_indices,
        symmetric_one_indices,
        symmetric_zero_indices
    ], axis=0)

    return modified_tensor, all_test_indices


def update_U(X1, Y1, D, V, W, si_weight, Ui, Qi, UD_penalty, F):
    ele1 = khatri_rao(V, D)
    ele2 = khatri_rao(W, D)
    term1 = 2 * x_weight * X1 @ ele1 + 2 * y_weight * Y1 @ ele2
    term2 = np.zeros((num_drug, rank), dtype=float)
    for i in range(num_si):
        term2 += 2 * si_weight[i] * Ui[i] @ Qi[i]
    term3 = UD_penalty * D + F
    first_term = term1 + term2 + term3

    term4 = 2 * x_weight * ele1.T @ ele1 + 2 * y_weight * ele2.T @ ele2
    term5 = np.zeros((rank, rank), dtype=float)
    for i in range(num_si):
        term5 += 2 * si_weight[i] * np.eye(rank)
    term6 = UD_penalty * np.eye(rank)
    second_term = term4 + term5 + term6
    ans = first_term @ np.linalg.inv(second_term)
    return ans


def update_D(X2, Y2, U, V, W, UD_penalty, F):
    ele1 = khatri_rao(V, U)
    ele2 = khatri_rao(W, U)
    first_term = 2 * x_weight * X2 @ ele1 + 2 * y_weight * Y2 @ ele2 - F + UD_penalty * U
    second_term = 2 * x_weight * ele1.T @ ele1 + 2 * y_weight * ele2.T @ ele2 + UD_penalty * np.eye(rank)
    ans = first_term @ np.linalg.inv(second_term)
    return ans


def update_V(X3, U, D):
    ele1 = khatri_rao(D, U)
    first_term = X3 @ ele1
    second_term = ele1.T @ ele1
    ans = first_term @ np.linalg.inv(second_term)
    return ans


def update_W(Y3, U, D):
    ele1 = khatri_rao(D, U)
    first_term = Y3 @ ele1
    second_term = ele1.T @ ele1
    ans = first_term @ np.linalg.inv(second_term)
    return ans


def update_Ui(Sa_i, C_i, U, Y_i, Q_i, SI_penalty, alpha_i):
    first_term = 2 * alpha_i * Sa_i.T @ C_i + 2 * alpha_i * U @ Q_i.T + SI_penalty * C_i + Y_i
    second_term = 2 * alpha_i * C_i.T @ C_i + 2 * alpha_i * Q_i @ Q_i.T + SI_penalty * np.eye(rank)
    ans = first_term @ np.linalg.inv(second_term)
    return ans


def update_Ci(U_i, Sa_i, Y_i, SI_penalty, alpha_i):
    first_term = 2 * alpha_i * Sa_i @ U_i + SI_penalty * U_i - Y_i
    second_term = 2 * alpha_i * U_i.T @ U_i + SI_penalty * np.eye(rank)
    ans = first_term @ np.linalg.inv(second_term)
    return ans


if __name__ == '__main__':
    args = parse_arguments()
    tol = args.tol
    # basic parameter of input data
    non_binary = args.nb
    x_name = args.x_data
    y_name = args.y_data
    test_index_rnd_seed = args.test_rnd_seed
    rnd_seeds = args.rnd_seeds
    rank = args.rank
    all_drugs = np.load(f'../data/{x_name}_{y_name}_all_drugs.npy')
    num_drug = len(all_drugs)
    si = args.si
    si_weight = args.si_weight
    x_weight = args.x_weight
    y_weight = args.y_weight
    num_si = len(si)
    missing_rate = args.missing_rate
    UD_penalty = args.UDpenalty
    SI_penalty = args.SIpenalty
    penalty_multiplier = args.penalty_multiplier
    max_penalty = args.max_penalty
    lr = args.lr
    num_iteration = args.train_iter

    # the root directery to read the data
    base_dir = args.base_dir

    # the directory and file name we are going to save the losses and result
    save_name = args.save_name
    for weight in [x_weight, y_weight] + si_weight:
        save_name += str(weight) + '_'

    # the directory and file name we are going to save the losses and result
    if non_binary:
        save_name = f'{x_name}_{y_name}_' + save_name + 'nb'
    else:
        save_name = f'{x_name}_{y_name}_' + save_name + 'b'

    print(f'is the model using non_binary data: {non_binary}')
    print(f"save name: {save_name}")

    save_dir = args.save_dir
    full_save_dir = save_dir + save_name

    real_tensor_x, real_tensor_y = load_tensor_x_y(args)
    num_disease = real_tensor_x.shape[2]
    num_ddi = real_tensor_y.shape[2]

    for rnd_seed in rnd_seeds:
        np.random.seed(rnd_seed)

        # generate the test tensor and save the indicies
        tensor_x, x_test_indices = generate_test_tensor(real_tensor_x, args)
        tensor_y, y_test_indices = generate_test_tensor(real_tensor_y, args)

        original_tensor_x = tensor_x.copy()
        original_tensor_y = tensor_y.copy()

        # create the complement weight matrix
        complement_X = np.zeros(real_tensor_x.shape, dtype=float)
        X_i, X_j, X_k = x_test_indices[:, 0], x_test_indices[:, 1], x_test_indices[:, 2]
        complement_X[X_i, X_j, X_k] = 1.0

        complement_Y = np.zeros(real_tensor_y.shape, dtype=float)
        Y_i, Y_j, Y_k = y_test_indices[:, 0], y_test_indices[:, 1], y_test_indices[:, 2]
        complement_Y[Y_i, Y_j, Y_k] = 1.0

        # load up the side information
        Sa = load_si(args)

        numbers = []
        print(f'norm of tensor X: {np.sum(tensor_x)}')
        numbers.append(np.sum(tensor_x))
        print(f'norm of tensor Y: {np.sum(tensor_y)}')
        numbers.append(np.sum(tensor_y))

        for i, sai in enumerate(Sa):
            print(f'norm of {i} th side info: {np.sum(sai)}')
            numbers.append(np.sum(sai))

        values = np.array(numbers)  # convert to NumPy array
        print(values)
        weights = 1 / values  # element-wise reciprocal
        weights /= np.sum(weights)  # normalize to sum to 1

        print(weights)

        # initial value of lagragian multiplier
        F = np.zeros((num_drug, rank), dtype=float)

        Y = []
        for si_i in range(num_si):
            Y.append(np.zeros((num_drug, rank), dtype=float))

        # initialize U, V, W, Ui, and D and Ci
        U = np.random.rand(num_drug, rank)
        D = np.random.rand(num_drug, rank)
        V = np.random.rand(num_disease, rank)
        W = np.random.rand(num_ddi, rank)
        Ui = []
        Ci = []

        for i in range(num_si):
            U_i = np.random.rand(num_drug, rank)
            Ui.append(U_i)
            C_i = np.random.rand(num_drug, rank)
            Ci.append(C_i)

        # Early stopping parameters
        patience = args.tolerance
        no_improve_counter = 0
        last_value = float('inf')

        # save the losses and stopping val
        x_losses = []
        y_losses = []
        x_stopping_vals = []
        y_stopping_vals = []
        stopping_vals = []
        f_vals = []
        lag_vals = []
        x_stop_numerators = []
        x_stop_denominators = []
        y_stop_numerators = []
        y_stop_denominators = []

        # start the optimizaiton
        for iter in range(num_iteration):
            Qi = []
            for i in range(num_si):
                Qi.append(create_scale_matrix(Ui[i]))

            U_new = update_U(mode_n_matricization(tensor_x, 1), mode_n_matricization(tensor_y, 1),
                             D, V, W, si_weight, Ui, Qi, UD_penalty, F)
            U_new = np.maximum(U_new, 0)

            D_new = update_D(mode_n_matricization(tensor_x, 2), mode_n_matricization(tensor_y, 2),
                             U_new, V, W, UD_penalty, F)
            D_new = np.maximum(D_new, 0)

            V_new = update_V(mode_n_matricization(tensor_x, 3), U_new, D_new)
            V_new = np.maximum(V_new, 0)

            W_new = update_W(mode_n_matricization(tensor_y, 3), U_new, D_new)
            W_new = np.maximum(W_new, 0)

            F_new = F + UD_penalty * (D_new - U_new)

            Ui_new = []
            Ci_new = []
            Y_new = []
            for si_i in range(num_si):
                Ui_new_i = update_Ui(Sa[si_i], Ci[si_i], U_new, Y[si_i], Qi[si_i], SI_penalty, si_weight[si_i])
                Ui_new_i = np.maximum(Ui_new_i, 0)
                Ui_new.append(Ui_new_i)

                Ci_new_i = update_Ci(Ui_new[si_i], Sa[si_i], Y[si_i], SI_penalty, si_weight[si_i])
                Ci_new_i = np.maximum(Ci_new_i, 0)
                Ci_new.append(Ci_new_i)
                Y_new.append(Y[si_i] + SI_penalty * (Ci_new[si_i] - Ui_new[si_i]))

            # the new penalty values
            new_UD_penalty = np.minimum(UD_penalty * penalty_multiplier, max_penalty)
            new_SI_penalty = np.minimum(SI_penalty * penalty_multiplier, max_penalty)

            # reconstruction loss
            error_X = real_tensor_x - resemble_tensor(U_new, D_new, V_new)
            masked_error_X = complement_X * (error_X ** 2)
            loss_X = np.sum(masked_error_X)

            error_Y = real_tensor_y - resemble_tensor(U_new, D_new, W_new)
            masked_error_Y = complement_Y * (error_Y ** 2)
            loss_Y= np.sum(masked_error_Y)

            # update the tensor
            new_X = original_tensor_x + complement_X * resemble_tensor(U_new, D_new, V_new)
            new_Y = original_tensor_y + complement_Y * resemble_tensor(U_new, D_new, W_new)
            x_stopping_val = np.linalg.norm(new_X - tensor_x) / np.linalg.norm(tensor_x)
            y_stopping_val = np.linalg.norm(new_Y - tensor_y) / np.linalg.norm(tensor_y)
            overall_stopping = x_stopping_val + y_stopping_val
            f_val, lag = lagrangian(U_new, D_new, V_new, W_new, Ci_new, Ui_new, Qi, F_new, Y_new, new_UD_penalty, new_SI_penalty)

            x_numerator = np.linalg.norm(new_X - tensor_x)
            x_denominator = np.linalg.norm(tensor_x)
            x_stop_numerators.append(x_numerator)
            x_stop_denominators.append(x_denominator)

            y_numerator = np.linalg.norm(new_Y - tensor_y)
            y_denominator = np.linalg.norm(tensor_y)
            y_stop_numerators.append(y_numerator)
            y_stop_denominators.append(y_denominator)

            x_losses.append(loss_X)
            y_losses.append(loss_Y)
            x_stopping_vals.append(x_stopping_val)
            y_stopping_vals.append(y_stopping_val)
            stopping_vals.append(overall_stopping)
            f_vals.append(f_val)
            lag_vals.append(lag)

            print(f"iteration: {iter},"
                  f" loss X: {loss_X},"
                  f" loss_Y: {loss_Y},"
                  f" X_improvement: {x_stopping_val},"
                  f" Y_improvement: {y_stopping_val},"
                  f" overall_stopping: {overall_stopping},"
                  f" f: {f_val},"
                  f" lag: {lag}")

            current_value = loss_X + loss_Y

            if current_value < last_value - 0.000000001:
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            last_value = current_value

            if no_improve_counter >= patience or (x_stopping_val + y_stopping_val) < tol:
                print(f"early stopping triggered at iteration {iter} due to no improvement on losses")
                break

            # update the value for the next iteration
            tensor_x = new_X
            tensor_y = new_Y
            U = U_new
            D = D_new
            V = V_new
            W = W_new
            Ui = Ui_new
            Ci = Ci_new
            F = F_new
            Y = Y_new

            # adjust the penalty
            UD_penalty = new_UD_penalty
            SI_penalty = new_SI_penalty

        # save the loss curve
        loss_dic = {
            'x_loss': x_losses,
            'y_loss': y_losses,
            'x_stopping_val': x_stopping_vals,
            'y_stopping_val': y_stopping_vals,
            'overall_stopping_val': stopping_vals,
            'f_val': f_vals,
            'lag_Val': lag_vals,
            'x_stop_numerator': x_stop_numerators,
            'x_stop_denominator': x_stop_denominators,
            'y_stop_numerator': y_stop_numerators,
            'y_stop_denominator': y_stop_denominators
        }
        loss_df = pd.DataFrame(loss_dic)
        loss_save_dir = full_save_dir + f'_losses_{rnd_seed}.csv'
        loss_df.to_csv(loss_save_dir, index=False)
        print(f'saving the loss to {loss_save_dir}')

        pred_x = resemble_tensor(U, D, V)
        pred_y = resemble_tensor(U, D, W)

        # save the result of the testing columns
        x_result, y_result = result_to_csv(real_tensor_x, real_tensor_y, pred_x, pred_y, x_test_indices, y_test_indices,
                                           full_save_dir, rnd_seed, rank)
