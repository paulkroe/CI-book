import numpy as np
import torch as T
import wandb
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt

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
            idxs  = self.indices_by_label[lbl]
            choice = np.random.choice(idxs)
            emb    = self.embeddings[choice]
            emb_t  = T.from_numpy(emb).to(self.device)
            out.append(emb_t)
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
        self.evaluating = evaluating
        embedding_npz = "dat/embeddings_epoch30_acc45.59.npz"
        super().__init__(mode)
        self.device = device or T.device('cpu')
        self.sampler = EmbeddingSampler(embedding_npz, device=self.device)
        D = self.sampler.embeddings.shape[1]

        # SCM vars: EMB (real ℝᵈ), E ∈ {±1}, Y ∈ {±1}
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
        do = do or {}
        u_i = exog["u_i"]   # shape (n,)
        u_e = exog["u_e"]   # shape (n,)
        n   = len(u_i)

        # ——— 1) Determine which “class bits” to use ———
        # If there’s a do-intervention on EMB, use that; otherwise use the exogenous draw.
        if "EMB" in do:
            cls_arr = do["EMB"].squeeze().cpu().numpy().astype(int)
        else:
            cls_arr = u_i.copy()

        # ——— 2) Sample embeddings based on cls_arr ———
        emb_list = self.sampler(cls_arr.tolist())
        EMB      = T.stack(emb_list, dim=0).to(self.device)  # (n, D)

        # ——— 3) Compute “industrial” from cls_arr ———
        industrial = np.isin(cls_arr, [0,1,8,9]).astype(int)

        # ——— 4) Compute E: either from do or industrial⊕u_e ———
        if "E" in do:
            E_bits = (do["E"].squeeze().cpu().numpy() > 0).astype(int)
        else:
            E_bits = np.logical_xor(industrial, u_e).astype(int)

        # ——— 5) Compute Y deterministically ———
        # Y=1 exactly when industrial == E
        Y_bits = (industrial == E_bits).astype(int)

        # ——— 6) Remap {0→–1, 1→+1} and package tensors ———
        E_bits = np.where(E_bits == 0, -1, +1)
        Y_bits = np.where(Y_bits == 0, -1, +1)

        E = T.from_numpy(E_bits).float().unsqueeze(1).to(self.device)
        Y = T.from_numpy(Y_bits).float().unsqueeze(1).to(self.device)

        return {"EMB": EMB, "E": E, "Y": Y}

    def generate_samples(self, n: int, obs: dict = None, do: dict = None):
        obs = obs or {}
        do  = do  or {}

        if obs:
            # only E can be observed here
            max_bs     = 5 * n
            exog_match = []
            while len(exog_match) < n:
                exog  = self._sample_exogenous(max_bs)
                batch = self._compute_from_exogenous(exog)
                mask  = np.ones(max_bs, dtype=bool)
                for key, tensor in obs.items():
                    if key != 'E':
                        raise ValueError("Can only observe 'E'")
                    target = tensor.squeeze().cpu().numpy().ravel()[0]
                    vals   = batch[key].squeeze().cpu().numpy()
                    mask  &= (vals == target)
                idxs = np.nonzero(mask)[0]
                for i in idxs:
                    exog_match.append({k: v[i] for k, v in exog.items()})
                    if len(exog_match) == n:
                        break
                if not idxs.size:
                    raise ValueError("No samples match obs")
            exog = {k: np.stack([em[k] for em in exog_match], axis=0)
                   for k in exog_match[0]}
        else:
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
        If `model` is provided, we sample embeddings for each class c and call
        model.forward(n=m, do={'EMB': embeddings}, evaluating=evaluating).
        Otherwise we call self.generate_samples(n=m, do={'EMB': class_labels}).
        Prints each probability and returns their average as a tensor.
        """
        results = []
        for c in range(10):
            name = f"P(Y=1|do-EMB={c})"

            if model is not None:
                print("Estiamte: ")
                # 1) build a batch of actual embeddings for class c
                emb_list   = self.sampler([c] * m)                   # list of length m
                emb_batch  = T.stack(emb_list, dim=0).to(self.device)  # (m, D)
                do_dict    = {"EMB": emb_batch}

                # 2) forward pass
                out = model.forward(n=m, do=do_dict, evaluating=evaluating)
                # assume out is either a dict with 'Y' or directly Y tensor
                ys = out["Y"] if isinstance(out, dict) else out

            else:
                print("Ground Truth: ")
                # fall back to sampling via SCMDataGenerator
                labels     = T.full((m,1), c, dtype=T.long, device=self.device)
                do_dict    = {"EMB": labels}
                batch      = self.generate_samples(n=m, do=do_dict)
                ys         = batch["Y"]

            # 3) compute probability
            prob = float((ys >= 0).float().mean().item())
            results.append((name, prob))

            # 4) optional logging and printing
            if log:
                wandb.log({name: prob})
            print(f"{name} = {prob:.3f}")

        # 5) return average probability
        avg = sum(p for _, p in results) / len(results)
        return T.tensor(avg, device=self.device)


# Example usage:
if __name__ == "__main__":
    gen = CIFAR10EmbeddingDataGenerator(
        embedding_npz="dat/embeddings_epoch30_acc45.59.npz",
        mode="sampling",
        device=T.device("cpu")
    )
    # draw 16 samples
    samples = gen.generate_samples(n=16)
    print(samples)
    # compute all P(Y=1|do...)
    estimates = gen.calculate_query(model=None, m=10000)
    for name, prob in estimates:
        print(f"{name} = {prob:.3f}")