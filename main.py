import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# ==================== CONFIG ====================
EMBED_DIM = 768          # BGE-base-en-v1.5 output dimension
N_ENCODERS = 2           # E: parallel simulated quantum encoders
QUBITS_PER_ENCODER = 4   # Q per encoder  →  latent_dim = E * Q = 8
N_LAYERS_ENCODER = 1     # variational layers inside each sQE
N_REUPLOADS = 3          # R: data re-uploading repetitions in the PQC
N_LAYERS_PER_REUPLOAD = 1  # M: entangling layers per re-upload block
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.01


# ==================== QUANTUM COMPONENTS ====================

def _make_encoder_qlayer(qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """Factory: one sQE QNode as a TorchLayer (amplitude encoding + variational circuit)."""
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(qubits), normalize=True)
        qml.StronglyEntanglingLayers(weights, wires=range(qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, qubits, 3)})


def _make_pqc_qlayer(n_qubits: int, n_reuploads: int, n_layers: int) -> qml.qnn.TorchLayer:
    """Factory: re-uploading PQC QNode as a TorchLayer (angle encoding + variational circuit)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for r in range(n_reuploads):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[r], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_reuploads, n_layers, n_qubits, 3)})


class MultiEncoderDR(nn.Module):
    """Simulated Quantum Encoder (sQE): classical projection → amplitude encoding
    → parameterized circuit → Pauli-Z measurements on all qubits.
    Uses E parallel encoders each with Q qubits; outputs E*Q latent features."""

    def __init__(self, embed_dim: int, n_encoders: int, qubits_per_encoder: int,
                 n_layers_encoder: int = 1):
        super().__init__()
        self.n_encoders = n_encoders
        self.latent_dim = n_encoders * qubits_per_encoder

        # Project embed_dim → 2^Q amplitude vector for each encoder
        self.projectors = nn.ModuleList([
            nn.Linear(embed_dim, 2 ** qubits_per_encoder, bias=False)
            for _ in range(n_encoders)
        ])
        self.qlayers = nn.ModuleList([
            _make_encoder_qlayer(qubits_per_encoder, n_layers_encoder)
            for _ in range(n_encoders)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, embed_dim)
        latents = []
        for i in range(self.n_encoders):
            amps = self.projectors[i](x)       # (B, 2^Q)
            latents.append(self.qlayers[i](amps))  # (B, Q)
        return torch.cat(latents, dim=-1)      # (B, latent_dim)


class ReuploadingPQC(nn.Module):
    """Re-uploading PQC: angle-encodes the latent vector R times interleaved with
    variational layers. Measures all qubits → (B, n_qubits) output."""

    def __init__(self, n_qubits: int, n_reuploads: int = 3, n_layers: int = 1):
        super().__init__()
        self.qlayer = _make_pqc_qlayer(n_qubits, n_reuploads, n_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # (B, n_qubits)
        return self.qlayer(latent)


class HybridQuantumHead(nn.Module):
    """Full hybrid classification head (paper Fig. 1):
      embed_dim  →  sQE (multi-encoder)  →  latent_dim
               →  re-uploading PQC      →  Qc measurements
               →  linear               →  num_classes logits
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        n_encoders: int = N_ENCODERS,
        qubits_per_encoder: int = QUBITS_PER_ENCODER,
        n_layers_encoder: int = N_LAYERS_ENCODER,
        n_reuploads: int = N_REUPLOADS,
        n_layers_per_reupload: int = N_LAYERS_PER_REUPLOAD,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        latent_dim = n_encoders * qubits_per_encoder

        self.encoder = MultiEncoderDR(embed_dim, n_encoders, qubits_per_encoder, n_layers_encoder)
        self.pqc = ReuploadingPQC(latent_dim, n_reuploads, n_layers_per_reupload)
        # nn.Linear includes a bias term, matching the paper's (Qc+1) → k mapping
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)       # (B, latent_dim)
        q_out = self.pqc(latent)       # (B, latent_dim)
        return self.classifier(q_out)  # (B, num_classes)


# ==================== TRAINING ====================

def train(x_train: np.ndarray, y_train: np.ndarray,
          x_test: np.ndarray, y_test: np.ndarray) -> HybridQuantumHead:

    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    x_te = torch.tensor(x_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    model = HybridQuantumHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel trainable parameters: {n_params}")
    print(f"Training {EPOCHS} epochs | batch {BATCH_SIZE} | lr {LR}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(x_te).argmax(dim=-1).numpy()
            acc = accuracy_score(y_te.numpy(), preds)
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss {avg_loss:.4f} | val acc {acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds = model(x_te).argmax(dim=-1).numpy()
    final_acc = accuracy_score(y_te.numpy(), preds)
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    return model
