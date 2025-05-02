import numpy as np
import torch as T
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt

# constants for grading / evaluation
ground_truth_0 = 0.46   # P(Y=1|industrial,do-EMB=c∉{0,1,8,9})
ground_truth_1 = 0.54   # P(Y=1|industrial,do-EMB=c∈{0,1,8,9})

class EmbeddingSampler:
    def __init__(self, npz_path: str, device: T.device = None):
        data = np.load(npz_path)
        self.embeddings = data['embeddings']  # shape [N, D]
        self.labels     = data['labels']      # shape [N]
        self.device     = device or T.device('cpu')

        # build index lists per label
        unique_labels = np.unique(self.labels)
        self.indices_by_label = {
            int(lbl): np.where(self.labels == lbl)[0]
            for lbl in unique_labels
        }

    def __call__(self, requested: list[int]) -> list[T.Tensor]:
        """
        For each class in `requested`, sample one random embedding from
        that class and return as a torch.Tensor.
        """
        out = []
        for lbl in requested:
            lbl = int(lbl)  # allow Tensor inputs
            idxs = self.indices_by_label[lbl]
            choice = np.random.choice(idxs)
            emb = self.embeddings[choice]          # numpy [D]
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

        # load your pretrained embedding file
        embedding_npz = "path/to/your/embedding_file.npz"
        self.sampler  = EmbeddingSampler(embedding_npz, device=self.device)
        D             = self.sampler.embeddings.shape[1]

        # define variable sizes/types for the SCM framework
        self.v_size = { "EMB": D, "E": 1, "Y": 1 }
        self.v_type = {
            "EMB": sdt.REAL,
            "E":   sdt.BINARY_ONES,
            "Y":   sdt.BINARY_ONES
        }
        self.cg = "path/to/your/graph"

    def _sample_exogenous(self, n: int):
        """
        [STEP 1 of SCM]
        Sample n i.i.d. draws of exogenous noise
        Return a dict: {"u_i": array(shape n,), "u_e": array(shape n,)}
        """
        # TODO:
        raise NotImplementedError

    def _compute_from_exogenous(self, exog: dict, do: dict = None):
        """
        [STEP 2 of SCM]
        Given your exogenous draws and an optional intervention `do`,
        implement the structural equations:

        1) Choose which class to use for embedding:
           cls_for_emb = do['EMB'] if present, else exog['u_i']  
        2) Sample embeddings: 
           EMB = stack( self.sampler(cls_for_emb) )  
        3) Compute 'industrial' from the ORIGINAL U_I:
           industrial_uI = 1 if exog['u_i'] ∈ {0,1,8,9} else 0  
        4) Compute environment E:
           E_bits = industrial_uI XOR exog['u_e']  
        5) Compute compatibility Y:
           industrial_do = 1 if cls_for_emb ∈ {0,1,8,9} else 0  
           Y_bits        = 1 if industrial_do == E_bits else 0  
        6) Remap bits {0→−1, 1→+1} and wrap into tensors.

        Return dict {"EMB": EMB_tensor,
                     "E":   E_tensor,
                     "Y":   Y_tensor}
        """
        # TODO:
        raise NotImplementedError

    def generate_samples(self, n: int, obs: dict = None, do: dict = None):
        """
        Utility that samples n points via:
          exog = self._sample_exogenous(n)
          out  = self._compute_from_exogenous(exog, do)
        (We ignore obs completely in this exercise.)
        """
        exog = self._sample_exogenous(n)
        data = self._compute_from_exogenous(exog, do=do or {})
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
        [STEP 3: Query Estimation]
        For each class c in 0..9:
          1) build a do‐intervention on EMB=c
          2) if model given:
               - sample real embeddings for class c
               - feed into model.forward(n=m, do={'EMB': embeddings}, evaluating)
               - collect output Y
             else: (this is used to estimate the query using the true SCM)
               - call self.generate_samples(n=m, do={'EMB':c})
               - collect Y
          3) estimate P(Y=1) by mean( Y_bits ≥ 0 )
        Finally compute total_error against ground_truth_1 or ground_truth_0
        and return it as a tensor.
        """
        # TODO:
        raise NotImplementedError