"""
Concept Bottleneck Model (CBM) for derm7pt.

    image features x  --g-->  concepts c_hat (7 sigmoid probs)  --f-->  diagnosis y_hat (5 classes)

The label head f ONLY sees the 7 concept probabilities -> this is the bottleneck.
Training regimes (Koh et al., 2020, "Concept Bottleneck Models"):
    independent : g trained on x -> c ; f trained on TRUE c -> y      (separately)
    sequential  : g trained on x -> c ; f trained on PREDICTED c_hat -> y (g frozen)
    joint       : g and f trained together with  CE(y) + lambda * BCE(c)
Plus a black-box baseline (x -> y directly, no concepts) for reference.

Model selection / early stopping uses the validation split only; the test split is
only used in `evaluate` and `intervention_curve`.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


# --------------------------------------------------------------------------- model
class CBM(nn.Module):
    def __init__(self, n_features, n_concepts, n_classes):
        super().__init__()
        self.g = nn.Linear(n_features, n_concepts)   # concept predictor  x -> c
        self.f = nn.Linear(n_concepts, n_classes)    # label predictor    c -> y (linear = interpretable)

    def concept_logits(self, x):
        return self.g(x)

    def forward(self, x, c_override=None, mask=None):
        """Returns (concept_logits, concept_probs, class_logits).
        Intervention: where mask==1, the predicted concept prob is replaced by c_override."""
        c_logits = self.g(x)
        c_prob = torch.sigmoid(c_logits)
        if c_override is not None:
            c_prob = torch.where(mask.bool(), c_override, c_prob)
        return c_logits, c_prob, self.f(c_prob)


# --------------------------------------------------------------------------- helpers
def _t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def class_weights(y, n_classes):
    """Inverse-frequency class weights (training set), normalised to mean 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1)
    return _t(w / w.mean())


def concept_pos_weight(c):
    """pos_weight for BCE so rare concepts (e.g. vascular structures, 6%) are not ignored."""
    p = c.mean(0).clip(1e-3, 1 - 1e-3)
    return _t((1 - p) / p)


def _fit(module, train_loss, val_score, epochs=5000, lr=5e-3, wd=1e-3, patience=300):
    """Full-batch Adam on `module`'s parameters; restores the parameters with the best
    validation score (higher = better) = early stopping on the validation split.
    We use the NEGATIVE validation LOSS as score: it is smooth, whereas balanced accuracy
    jumps around and made early stopping pick near-untrained models."""
    modules = [module]
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=wd)
    best, best_state, since = -np.inf, None, 0
    for ep in range(epochs):
        for m in modules:
            m.train()
        opt.zero_grad()
        loss = train_loss()
        loss.backward()
        opt.step()
        for m in modules:
            m.eval()
        with torch.no_grad():
            s = val_score()
        if s > best + 1e-5:
            best, since = s, 0
            best_state = [copy.deepcopy(m.state_dict()) for m in modules]
        else:
            since += 1
            if since > patience:
                break
    for m, st in zip(modules, best_state):
        m.load_state_dict(st)
    return best


# --------------------------------------------------------------------------- training regimes
def train_concept_head(model, D, cw_pos=None):
    """Train g on x -> c (used by independent and sequential). Early stopping: val BCE.
    No pos_weight by default: f consumes the concept PROBABILITIES, so they should stay calibrated."""
    Xtr, Ctr, Xva, Cva = D['Xtr'], D['Ctr'], D['Xva'], D['Cva']
    loss = lambda: F.binary_cross_entropy_with_logits(model.g(Xtr), Ctr, pos_weight=cw_pos)
    score = lambda: -F.binary_cross_entropy_with_logits(model.g(Xva), Cva, pos_weight=cw_pos).item()
    _fit(model.g, loss, score)


def train_label_head(model, Ctr_input, D, cw_cls, val_concepts):
    """Train f on given concept inputs -> y. Early stopping: val cross-entropy on `val_concepts`
    (TRUE val concepts for independent, PREDICTED val concepts for sequential)."""
    Ytr, Yva = D['Ytr'], D['Yva']
    loss = lambda: F.cross_entropy(model.f(Ctr_input), Ytr, weight=cw_cls)
    score = lambda: -F.cross_entropy(model.f(val_concepts), Yva, weight=cw_cls).item()
    _fit(model.f, loss, score)


def train_independent(D, seed=0):
    torch.manual_seed(seed)
    m = CBM(D['Xtr'].shape[1], D['Ctr'].shape[1], D['n_classes'])
    train_concept_head(m, D)
    # f is trained AND early-stopped on TRUE concepts; it only meets predicted concepts at test time
    train_label_head(m, D['Ctr'], D, D['cw_cls'], D['Cva'])
    return m


def train_sequential(D, seed=0, concept_model=None):
    torch.manual_seed(seed)
    m = CBM(D['Xtr'].shape[1], D['Ctr'].shape[1], D['n_classes'])
    if concept_model is not None:                       # reuse the same g as independent (fair comparison)
        m.g.load_state_dict(concept_model.g.state_dict())
    else:
        train_concept_head(m, D)
    with torch.no_grad():
        tr_c = torch.sigmoid(m.g(D['Xtr']))             # f trained on PREDICTED train concepts
        val_c = torch.sigmoid(m.g(D['Xva']))
    train_label_head(m, tr_c, D, D['cw_cls'], val_c)
    return m


def train_joint(D, lam=1.0, seed=0):
    torch.manual_seed(seed)
    m = CBM(D['Xtr'].shape[1], D['Ctr'].shape[1], D['n_classes'])

    def joint_loss(split):
        c_logits, _, y_logits = m(D['X' + split])
        return (F.cross_entropy(y_logits, D['Y' + split], weight=D['cw_cls'])
                + lam * F.binary_cross_entropy_with_logits(c_logits, D['C' + split]))

    loss = lambda: joint_loss('tr')
    score = lambda: -joint_loss('va').item()
    _fit(m, loss, score)
    return m


class BlackBox(nn.Module):
    """Baseline without a bottleneck: x -> y directly."""
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.h = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.h(x)


def train_blackbox(D, seed=0):
    torch.manual_seed(seed)
    m = BlackBox(D['Xtr'].shape[1], D['n_classes'])
    loss = lambda: F.cross_entropy(m(D['Xtr']), D['Ytr'], weight=D['cw_cls'])
    score = lambda: -F.cross_entropy(m(D['Xva']), D['Yva'], weight=D['cw_cls']).item()
    _fit(m, loss, score)
    return m


# --------------------------------------------------------------------------- data + evaluation
def make_data(X, C, Y, split_idx, n_classes):
    """Standardise features with TRAIN statistics only, and pack everything as tensors."""
    tr, va, te = split_idx['train'], split_idx['valid'], split_idx['test']
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    D = {'n_classes': n_classes}
    for name, idx in [('tr', tr), ('va', va), ('te', te)]:
        D['X' + name] = _t(Z[idx])
        D['C' + name] = _t(C[idx])
        D['Y' + name] = _t(Y[idx], torch.long)
    D['cw_cls'] = class_weights(Y[tr], n_classes)
    D['cw_pos'] = concept_pos_weight(C[tr])
    return D


def mean_auc(c_true, c_prob):
    aucs = [roc_auc_score(c_true[:, j], c_prob[:, j])
            for j in range(c_true.shape[1]) if 0 < c_true[:, j].sum() < len(c_true)]
    return float(np.mean(aucs))


def label_metrics(y_true, y_pred):
    return {'acc': accuracy_score(y_true, y_pred),
            'bal_acc': balanced_accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro')}


@torch.no_grad()
def evaluate(model, D, split='te'):
    X, C, Y = D['X' + split], D['C' + split].numpy(), D['Y' + split].numpy()
    model.eval()
    if isinstance(model, BlackBox):
        return label_metrics(Y, model(X).argmax(1).numpy())
    _, c_prob, y_logits = model(X)
    c_prob = c_prob.numpy()
    out = label_metrics(Y, y_logits.argmax(1).numpy())
    out['concept_acc'] = float(((c_prob > 0.5) == C).mean())
    out['concept_auc'] = mean_auc(C, c_prob)
    return out


@torch.no_grad()
def per_concept_auc(model, D, names, split='te'):
    C = D['C' + split].numpy()
    c_prob = torch.sigmoid(model.g(D['X' + split])).numpy()
    return {n: roc_auc_score(C[:, j], c_prob[:, j]) for j, n in enumerate(names)}


@torch.no_grad()
def intervention_curve(model, D, split='te', n_orders=20, seed=0, metric='bal_acc'):
    """Replace k randomly chosen predicted concepts with their TRUE values (k = 0..n_concepts),
    averaged over `n_orders` random concept orderings (as in the CBM paper)."""
    X, C, Y = D['X' + split], D['C' + split], D['Y' + split].numpy()
    k_max = C.shape[1]
    rng = np.random.default_rng(seed)
    scores = np.zeros((n_orders, k_max + 1))
    for r in range(n_orders):
        order = rng.permutation(k_max)
        for k in range(k_max + 1):
            mask = torch.zeros_like(C)
            mask[:, order[:k]] = 1
            y_pred = model(X, c_override=C, mask=mask)[2].argmax(1).numpy()
            scores[r, k] = label_metrics(Y, y_pred)[metric]
    return scores.mean(0), scores.std(0)