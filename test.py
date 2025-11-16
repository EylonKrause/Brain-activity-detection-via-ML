
import numpy as np
import torch
import scipy.io as sio
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# from google.colab import drive
# drive.flush_and_unmount()
# drive.mount('/content/drive/', force_remount=True)

# !ls /content/drive

# # we placed the files in our pc here, put it in comment so it won't interfere with your loading
# %cd "/content/drive/My Drive/שיעורים פרטיי/ג'וליה FMRI/עבודת הגשה סופית"

# %cd G:\My Drive\שיעורים פרטיים\ג'וליה FMRI\עבודת הגשה סופית

# %cd E:\Education\Former TA\שיעורים פרטיים\ג'וליה FMRI\Brain-activity-detection-via-ML\עבודת הגשה סופית

# rest_path_VIS = "/content/drive/MyDrive/data_rest_VIS.mat"
# rest_path_DAN = "/content/drive/MyDrive/data_rest_DAN.mat"
# rest_path_DMN = "/content/drive/MyDrive/data_rest_DMN.mat"

rest_path_VIS = r"E:\Education\Former TA\שיעורים פרטיים\ג'וליה FMRI\Brain-activity-detection-via-ML\עבודת הגשה סופית\data_rest_VIS.mat"
rest_path_DAN = r"E:\Education\Former TA\שיעורים פרטיים\ג'וליה FMRI\Brain-activity-detection-via-ML\עבודת הגשה סופית\data_rest_DAN.mat"
rest_path_DMN = r"E:\Education\Former TA\שיעורים פרטיים\ג'וליה FMRI\Brain-activity-detection-via-ML\עבודת הגשה סופית\data_rest_DMN.mat"

# rest_path_VIS = r"G:\My Drive\שיעורים פרטיים\ג'וליה FMRI\עבודת הגשה סופית\data_rest_VIS.mat"
# rest_path_DAN = r"G:\My Drive\שיעורים פרטיים\ג'וליה FMRI\עבודת הגשה סופית\data_rest_DAN.mat"
# rest_path_DMN = r"G:\My Drive\שיעורים פרטיים\ג'וליה FMRI\עבודת הגשה סופית\data_rest_DMN.mat"


# Load each file and extract the variables
rest_data_VIS = sio.loadmat(rest_path_VIS)['data_rest_VIS']
rest_data_DAN = sio.loadmat(rest_path_DAN)['data_rest_DAN']
rest_data_DMN = sio.loadmat(rest_path_DMN)['data_rest_DMN']

# Step 3: Check shapes
print("VIS rest shape:", rest_data_VIS.shape)
print("DAN rest shape:", rest_data_DAN.shape)
print("DMN rest shape:", rest_data_DMN.shape)


def extract_rest_features_labels(rest_data, var_name):
    """
    Extracts features and labels from resting-state fMRI data.

    Parameters:
        rest_data (np.ndarray): 4D array with shape (subjects, clips, time, regions)
        var_name (str): Just for labeling, not functionally used

    Returns:
        X_start (np.ndarray): Flattened features from first 5s [samples, features]
        Y_start (np.ndarray): Corresponding labels [samples]
        X_end (np.ndarray): Flattened features from last 5s [samples, features]
        Y_end (np.ndarray): Corresponding labels [samples]
    """
    X_start, Y_start = [], []
    X_end, Y_end = [], []

    for subj in range(rest_data.shape[0]):       # 170 subjects
        for clip in range(rest_data.shape[1]):   # 14 clips/rest periods
            segment = rest_data[subj, clip, :, :]  # shape (19, regions)

            start = segment[0:5, :]    # seconds 0–4
            end = segment[14:19, :]    # seconds 14–18

            X_start.append(start.flatten())
            Y_start.append(clip + 1)

            X_end.append(end.flatten())
            Y_end.append(clip + 1)

    return (
        np.array(X_start), np.array(Y_start),
        np.array(X_end), np.array(Y_end)
    )



# Use the previous function to extract features from resting-state data
X_VIS_rest_start, Y_VIS_rest_start, X_VIS_rest_end, Y_VIS_rest_end = extract_rest_features_labels(rest_data_VIS, "VIS")
X_DAN_rest_start, Y_DAN_rest_start, X_DAN_rest_end, Y_DAN_rest_end = extract_rest_features_labels(rest_data_DAN, "DAN")
X_DMN_rest_start, Y_DMN_rest_start, X_DMN_rest_end, Y_DMN_rest_end = extract_rest_features_labels(rest_data_DMN, "DMN")

# # Display the shapes
# (
#     X_VIS_rest_start.shape, X_VIS_rest_end.shape,
#     X_DAN_rest_start.shape, X_DAN_rest_end.shape,
#     X_DMN_rest_start.shape, X_DMN_rest_end.shape
# )

# ------------------------------------------------------------------#
#  Small utility classes implemented in pure‑Torch                  #
# ------------------------------------------------------------------#
class TorchStandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(0, keepdim=True)
        self.std_  = X.std (0, unbiased=False, keepdim=True).clamp(min=1e-8)
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class TorchLinearSVM(torch.nn.Module):
    """
    Linear SVM trained with hinge loss + L2 regularisation.
    Equivalent to LinearSVC(kernel='linear') for C=1/λ.
    """
    def __init__(self, n_features, C=1.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(n_features, 1))
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.C = C

    def forward(self, X):
        return X @ self.w + self.b          # shape (N,1)

    def fit(self, X, y, lr=1e-2, epochs=1000):
        y = y.view(-1, 1).float() * 2 - 1   # {0,1} → {-1,+1}
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            scores = self(X)
            hinge = torch.clamp(1 - y * scores, min=0)
            loss  = 0.5 * (self.w**2).sum() + self.C * hinge.mean()
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        return (self(X) > 0).long().view(-1)


class TorchLDA:
    def fit(self, X, y):
        self.classes_ = torch.unique(y)
        idx0, idx1   = (y == 0), (y == 1)
        self.m0_ = X[idx0].mean(0)
        self.m1_ = X[idx1].mean(0)
        S_w = torch.cov(X[idx0].T) * idx0.sum() + torch.cov(X[idx1].T) * idx1.sum()
        self.S_inv_ = torch.linalg.inv(S_w / (len(y) - 2))
        return self
    def _score(self, X, m):
        return (X @ self.S_inv_ @ m) - 0.5 * (m @ self.S_inv_ @ m)
    def predict(self, X):
        g0 = self._score(X, self.m0_)
        g1 = self._score(X, self.m1_)
        return (g1 > g0).long()


class TorchQDA:
    def fit(self, X, y):
        self.classes_ = torch.unique(y)
        self.means_ = []
        self.S_inv_ = []
        self.logdet_ = []
        for c in self.classes_:
            Xc = X[y == c]
            m  = Xc.mean(0)
            S  = torch.cov(Xc.T) + 1e-6 * torch.eye(X.shape[1], device=X.device)
            self.means_.append(m)
            self.S_inv_.append(torch.linalg.inv(S))
            self.logdet_.append(torch.logdet(S))
        return self
    def _score(self, X, k):
        m, S_inv, logdet = self.means_[k], self.S_inv_[k], self.logdet_[k]
        diff = X - m
        return -0.5 * ( (diff @ S_inv) * diff ).sum(1 ) - 0.5 * logdet
    def predict(self, X):
        g0 = self._score(X, 0)
        g1 = self._score(X, 1)
        return (g1 > g0).long()

# ------------------------------------------------------------------#
#  Generic LOOCV runner                                             #
# ------------------------------------------------------------------#
def _loo_predict(X, y, model_ctor, device):
    N      = X.shape[0]
    y_pred = torch.empty_like(y)
    for i in range(N):
        train_mask = torch.ones(N, dtype=torch.bool, device=device)
        train_mask[i] = False
        X_train, y_train = X[train_mask], y[train_mask]
        X_test          = X[~train_mask].unsqueeze(0)

        scaler = TorchStandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)

        model  = model_ctor(X_train.shape[1]) if callable(model_ctor) else model_ctor()
        model.fit(X_train, y_train)
        y_pred[i] = model.predict(X_test)
    return y_pred


def _run_loocv(X, Y, label, model_ctor, name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X = torch.as_tensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
                        dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.long, device=device)

    y_pred = _loo_predict(X, Y, model_ctor, device)
    acc    = (y_pred == Y).float().mean().item()
    print(f"{label}: {name} accuracy = {acc:.3f}")
    return acc, Y.cpu().tolist(), y_pred.cpu().tolist()

# ------------------------------------------------------------------#
#  Public API – exactly your old signatures                          #
# ------------------------------------------------------------------#
def run_svm_loocv_full(Xs, Ys, Xe, Ye, name=""):
    model_ctor = lambda nfeat: TorchLinearSVM(nfeat, C=1.0)
    acc_s, yT_s, yP_s = _run_loocv(Xs, Ys, "Start", model_ctor, "SVM")
    acc_e, yT_e, yP_e = _run_loocv(Xe, Ye, "End",   model_ctor, "SVM")
    return {'start': {'acc': acc_s, 'y_true': yT_s, 'y_pred': yP_s},
            'end':   {'acc': acc_e, 'y_true': yT_e, 'y_pred': yP_e}}

def run_lda_loocv_full(Xs, Ys, Xe, Ye, name=""):
    acc_s, yT_s, yP_s = _run_loocv(Xs, Ys, "Start", TorchLDA, "LDA")
    acc_e, yT_e, yP_e = _run_loocv(Xe, Ye, "End",   TorchLDA, "LDA")
    return {'start': {'acc': acc_s, 'y_true': yT_s, 'y_pred': yP_s},
            'end':   {'acc': acc_e, 'y_true': yT_e, 'y_pred': yP_e}}

def run_qda_loocv_full(Xs, Ys, Xe, Ye, name=""):
    acc_s, yT_s, yP_s = _run_loocv(Xs, Ys, "Start", TorchQDA, "QDA")
    acc_e, yT_e, yP_e = _run_loocv(Xe, Ye, "End",   TorchQDA, "QDA")
    return {'start': {'acc': acc_s, 'y_true': yT_s, 'y_pred': yP_s},
            'end':   {'acc': acc_e, 'y_true': yT_e, 'y_pred': yP_e}}


# knn_rest_VIS = run_knn_loocv_full(X_VIS_rest_start, Y_VIS_rest_start, X_VIS_rest_end, Y_VIS_rest_end, name="VIS Rest")
# knn_rest_DAN = run_knn_loocv_full(X_DAN_rest_start, Y_DAN_rest_start, X_DAN_rest_end, Y_DAN_rest_end, name="DAN Rest")
# knn_rest_DMN = run_knn_loocv_full(X_DMN_rest_start, Y_DMN_rest_start, X_DMN_rest_end, Y_DMN_rest_end, name="DMN Rest")


# svm_rest_VIS = run_svm_kfold_full(X_VIS_rest_start, Y_VIS_rest_start, X_VIS_rest_end, Y_VIS_rest_end, name="VIS Rest")
# svm_rest_DAN = run_svm_kfold_full(X_DAN_rest_start, Y_DAN_rest_start, X_DAN_rest_end, Y_DAN_rest_end, name="DAN Rest")
# svm_rest_DMN = run_svm_kfold_full(X_DMN_rest_start, Y_DMN_rest_start, X_DMN_rest_end, Y_DMN_rest_end, name="DMN Rest")

# LOO for section 2
svm_rest_VIS = run_svm_loocv_full(X_VIS_rest_start, Y_VIS_rest_start, X_VIS_rest_end, Y_VIS_rest_end, name="VIS Rest")
svm_rest_DAN = run_svm_loocv_full(X_DAN_rest_start, Y_DAN_rest_start, X_DAN_rest_end, Y_DAN_rest_end, name="DAN Rest")
svm_rest_DMN = run_svm_loocv_full(X_DMN_rest_start, Y_DMN_rest_start, X_DMN_rest_end, Y_DMN_rest_end, name="DMN Rest")

