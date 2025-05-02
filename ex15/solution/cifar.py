import numpy as np
import torch as T
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt

ground_truth_0 = 0.46
ground_truth_1 = 0.54 

class EmbeddingSampler:
    def __init__(self, npz_path: str, device: T.device = None):
        data = np.load(npz_path)
        self.embeddings = data['embeddings']  # shape [N, D]
        self.labels     = data['labels']      # shape [N]
        self.device     = device or T.device('cpu')

        unique_labels = np.unique(self.labels)
        self.indices_by_label = {
            int(lbl): np.where(self.labels == lbl)[0]
            for lbl in unique_labels
        }
        for lbl, idxs in self.indices_by_label.items():
            if len(idxs) == 0:
                raise ValueError(f"No embeddings for label {lbl}")

    def __call__(self, requested: list[int]) -> list[T.Tensor]:
        out = []
        for lbl in requested:
            if isinstance(lbl, T.Tensor):
                lbl = lbl.item()
            if lbl not in self.indices_by_label:
                raise KeyError(f"Unknown label {lbl}")
            idxs   = self.indices_by_label[lbl]
            choice = np.random.choice(idxs)
            emb    = self.embeddings[choice]
            out.append(T.from_numpy(emb).to(self.device))
        return out


class CIFAR10EmbeddingDataGenerator(SCMDataGenerator):
    def __init__(
        self,
        image_size: str = None,
        mode: str = "sampling",
        evaluating: bool = False,
        normalize: bool = False,
        device: T.device = None
    ):
        super().__init__(mode)
        self.evaluating = evaluating
        self.device     = device or T.device('cpu')

        embedding_npz = "dat/embeddings_epoch30_acc45.59.npz"
        self.sampler  = EmbeddingSampler(embedding_npz, device=self.device)
        D             = self.sampler.embeddings.shape[1]

        # SCM vars: EMB∈ℝᵈ, E∈{±1}, Y∈{±1}
        self.v_size = { "EMB": D, "E": 1, "Y": 1 }
        self.v_type = { "EMB": sdt.REAL,
                        "E":   sdt.BINARY_ONES,
                        "Y":   sdt.BINARY_ONES }
        self.cg = "cifar10"

    def _sample_exogenous(self, n: int):
        # U_I ∼ Uniform({0,…,9}), U_E ∼ Bernoulli(0.3)
        u_i = np.random.choice(10, size=n, p=[0.1]*10)
        u_e = (np.random.rand(n) < 0.3).astype(int)
        return {"u_i": u_i, "u_e": u_e}

    def _compute_from_exogenous(self, exog: dict, do: dict = None):
        do    = do or {}
        u_i   = exog["u_i"]   # shape (n,)
        u_e   = exog["u_e"]   # shape (n,)
        n     = len(u_i)

        # ——— 1) Which classes for sampling EMB? ———
        if "EMB" in do:
            cls_for_emb = do["EMB"].squeeze().cpu().numpy().astype(int)
        else:
            cls_for_emb = u_i.copy()

        # ——— 2) Sample your embeddings ———
        emb_list = self.sampler(cls_for_emb.tolist())
        EMB      = T.stack(emb_list, dim=0).to(self.device)  # (n, D)

        # ——— 3) industrial bit from exogenous U_I ———
        industrial_uI = np.isin(u_i, [0,1,8,9]).astype(int)

        # ——— 4) f_E = industrial(U_I) ⊕ U_E ———
        E_bits = np.logical_xor(industrial_uI, u_e).astype(int)

        # ——— 5) compatibility bit from the *intervened* class ———
        industrial_do = np.isin(cls_for_emb, [0,1,8,9]).astype(int)
        Y_bits        = (industrial_do == E_bits).astype(int)

        # ——— 6) remap to {−1,+1} and pack tensors ———
        E_bits = np.where(E_bits == 0, -1, +1)
        Y_bits = np.where(Y_bits == 0, -1, +1)

        E = T.from_numpy(E_bits).float().unsqueeze(1).to(self.device)
        Y = T.from_numpy(Y_bits).float().unsqueeze(1).to(self.device)

        return {"EMB": EMB, "E": E, "Y": Y}

    def generate_samples(self, n: int, obs: dict = None, do: dict = None):
        # ignore obs entirely; only exogenous + do on EMB
        do   = do or {}
        exog = self._sample_exogenous(n)
        data = self._compute_from_exogenous(exog, do=do)
        return {k: v.cpu() for k, v in data.items()}

    def calculate_query(
        self,
        model=None,
        tau=None,
        m: int = 10000,
        evaluating: bool = False,
        log: bool = False
    ):
        """
        Computes P(Y=1 | do(EMB=c)) for c=0..9.
        If `model` is provided, we sample real embeddings and call
        model.forward(n=m, do={'EMB': embeddings}, evaluating=evaluating).
        Otherwise we sample via the SCM.
        """
        results = []
        total_error = 0
        for c in range(10):
            name      = f"P(Y=1|do-EMB={c})"
            # build embeddings or class-labels
            if model is not None:
                emb_list  = self.sampler([c] * m)
                emb_batch = T.stack(emb_list, dim=0).to(self.device)
                do_dict   = {"EMB": emb_batch}
                out       = model.forward(n=m, do=do_dict, evaluating=evaluating)
                ys        = out["Y"] if isinstance(out, dict) else out
            else:
                labels    = T.full((m,1), c, dtype=T.long, device=self.device)
                do_dict   = {"EMB": labels}
                batch     = self.generate_samples(n=m, do=do_dict)
                ys        = batch["Y"]

            prob = float((ys >= 0).float().mean().item())
            results.append((name, prob))
            if log:
                import wandb
                wandb.log({name: prob})
            if model is not None:
                err = abs(prob - (ground_truth_0 if c in [0,1,8,9] else ground_truth_1))
                total_error += err
                print(f"Error {name}: {prob:.3f}")
            else:
                print(f"{name} = {prob:.3f}")

        return T.tensor(total_error, device=self.device)