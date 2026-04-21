#!/usr/bin/env python3
"""
Q11 Transformer — tehnika: Quantum Multi-Head Transformer (čisto kvantno)
(multi-head self-attention preko SWAP-test-a + kvantni FFN, bez klasičnog softmax-a i bez hibrida).

Arhitektura (nadogradnja nad Q10 single-head/single-layer):
  - Ulaz: Query stanje |Q⁽⁰⁾⟩ = amplitude-encoding freq_vector-a CELOG CSV-a (dim 2^nq).
  - Blokovi: CSV → B uzastopnih blokova, svaki |K_b⟩ = |V_b⟩ amplitude-encoding
    (fiksno kroz sve slojeve).
  - Po sloju l = 0..L-1:
      · H glava: h-ta glava koristi cikličko rotiran query
          |Q_h⁽ˡ⁾⟩ = roll(|Q⁽ˡ⁾⟩, h · Δ),  Δ = dim // H.
      · Attention težine po glavi (kvantno): w_{h,b} = |⟨Q_h⁽ˡ⁾|K_b⟩|²
        iz SWAP-test kola → egzaktni Statevector, P(anc=0) = (1+|⟨·|·⟩|²)/2.
      · p_h = Σ_b (w_{h,b}/Σ w_{h,·}) · |K_b|².
      · Multi-head mix: p⁽ˡ⁾ = (1/H) Σ_h p_h.
      · Kvantni FFN (čisto kvantno): amp_in = √p⁽ˡ⁾ (L2-renorm);
        PQC sloj = Ry(θ_k) + ring-CNOT, θ_k = π·p⁽ˡ⁾[k]·dim (deterministički);
        |Q⁽ˡ⁺¹⁾⟩ = √(|Statevector(PQC·|amp_in⟩)|²) (L2-renorm).
  - Izlaz: p⁽ᴸ⁻¹⁾ → bias_39 (mod-39) → NEXT (TOP-7).

Sve deterministički: seed=39; svi amp-ovi i parametri iz CELOG CSV-a.
Deterministička grid-optimizacija (nq, H, B, L) po meri cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_H = (2, 3)
GRID_B = (4, 5, 7)
GRID_L = (1, 2)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


def block_amps(H: np.ndarray, nq: int, B: int) -> List[np.ndarray]:
    n = H.shape[0]
    edges = np.linspace(0, n, B + 1, dtype=int)
    out: List[np.ndarray] = []
    for i in range(B):
        if edges[i + 1] <= edges[i]:
            out.append(amp_from_freq(np.zeros(N_MAX), nq))
        else:
            out.append(amp_from_freq(freq_vector(H[edges[i] : edges[i + 1]]), nq))
    return out


# =========================
# SWAP test — kvantno |⟨Q|K⟩|²
# =========================
def swap_test_overlap_sq(nq: int, amp_q: np.ndarray, amp_k: np.ndarray) -> float:
    total = 1 + 2 * nq
    qc = QuantumCircuit(total, name="swap_test")
    qc.append(StatePreparation(amp_q.tolist()), list(range(1, 1 + nq)))
    qc.append(StatePreparation(amp_k.tolist()), list(range(1 + nq, 1 + 2 * nq)))
    qc.h(0)
    for i in range(nq):
        qc.cswap(0, 1 + i, 1 + nq + i)
    qc.h(0)
    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    dim = 2 ** total
    p_anc0 = float(sum(p[i] for i in range(dim) if (i & 1) == 0))
    return float(max(0.0, 2.0 * p_anc0 - 1.0))


# =========================
# Kvantni FFN — PQC nad amp_in = √p_layer
# =========================
def ffn_update_amp(nq: int, p_layer: np.ndarray) -> np.ndarray:
    """amp_new = L2-renormalizovan √(|Statevector(PQC · |amp_in⟩)|²)."""
    dim = 2 ** nq
    amp_in = np.sqrt(np.maximum(p_layer, 0.0))
    n2 = float(np.linalg.norm(amp_in))
    amp_in = amp_in / n2 if n2 > 1e-18 else np.ones(dim) / np.sqrt(dim)

    thetas = np.array([float(np.pi * p_layer[k] * dim) for k in range(dim)], dtype=np.float64)
    qc = QuantumCircuit(nq, name="ffn")
    qc.append(StatePreparation(amp_in.tolist()), range(nq))
    for k in range(nq):
        qc.ry(float(thetas[k % dim]), k)
    for k in range(nq - 1):
        qc.cx(k, k + 1)
    if nq > 1:
        qc.cx(nq - 1, 0)

    sv = Statevector(qc)
    p_out = np.abs(sv.data) ** 2
    s = float(p_out.sum())
    p_out = p_out / s if s > 0 else p_out
    amp_out = np.sqrt(p_out)
    n2 = float(np.linalg.norm(amp_out))
    return amp_out / n2 if n2 > 1e-18 else np.ones(dim) / np.sqrt(dim)


# =========================
# Multi-Head self-attention po sloju
# =========================
def layer_probs(nq: int, amp_q: np.ndarray, amps_k: List[np.ndarray], H_heads: int) -> np.ndarray:
    dim = 2 ** nq
    delta = max(1, dim // max(1, H_heads))
    B = len(amps_k)

    p_sum = np.zeros(dim, dtype=np.float64)
    for h in range(H_heads):
        q_h = np.roll(amp_q, h * delta)
        # num. sigurnost: renorm
        n2 = float(np.linalg.norm(q_h))
        q_h = q_h / n2 if n2 > 1e-18 else amp_q
        w = np.array([swap_test_overlap_sq(nq, q_h, amps_k[b]) for b in range(B)], dtype=np.float64)
        s_w = float(w.sum())
        w = w / s_w if s_w > 1e-18 else np.ones(B) / B
        p_h = np.zeros(dim, dtype=np.float64)
        for b in range(B):
            p_h += float(w[b]) * (amps_k[b] ** 2)
        s_p = float(p_h.sum())
        p_h = p_h / s_p if s_p > 0 else p_h
        p_sum += p_h

    p_mean = p_sum / float(H_heads)
    s = float(p_mean.sum())
    return p_mean / s if s > 0 else p_mean


def transformer_probs(H_csv: np.ndarray, nq: int, H_heads: int, B: int, L: int) -> np.ndarray:
    f_csv = freq_vector(H_csv)
    amp_q = amp_from_freq(f_csv, nq)
    amps_k = block_amps(H_csv, nq, B)

    p_layer = None
    for _ in range(L):
        p_layer = layer_probs(nq, amp_q, amps_k, H_heads)
        amp_q = ffn_update_amp(nq, p_layer)
    return p_layer if p_layer is not None else (amp_q ** 2)


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, H, B, L)
# =========================
def optimize_hparams(H_csv: np.ndarray):
    f_csv = freq_vector(H_csv)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for Hh in GRID_H:
            for B in GRID_B:
                for L in GRID_L:
                    try:
                        p = transformer_probs(H_csv, nq, Hh, B, L)
                        b = bias_39(p)
                        score = cosine(b, f_csv_n)
                    except Exception:
                        continue
                    key = (score, -nq, -Hh, -B, -L)
                    if best is None or key > best[0]:
                        best = (key, dict(nq=nq, H=Hh, B=B, L=L, score=float(score)))
    return best[1] if best else None


def main() -> int:
    H_csv = load_rows(CSV_PATH)
    if H_csv.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q11 Transformer (Multi-Head QSAN + kvantni FFN): CSV:", CSV_PATH)
    print("redova:", H_csv.shape[0], "| seed:", SEED)

    best = optimize_hparams(H_csv)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| H (glava):", best["H"],
        "| B (blokova):", best["B"],
        "| L (slojeva):", best["L"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    p = transformer_probs(H_csv, best["nq"], best["H"], best["B"], best["L"])
    pred = pick_next_combination(p)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q11 Transformer (Multi-Head QSAN + kvantni FFN): CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 5 | H (glava): 3 | B (blokova): 7 | L (slojeva): 2 | cos(bias, freq_csv): 0.900349
predikcija NEXT: (7, 19, 22, 24, 27, 28, 31)
"""



"""
Q11_Transformer_Multi.py — tehnika: Quantum Multi-Head Transformer

Po sloju l:
  · H glava: Q_h = roll(Q, h·Δ) — per-head bazna rotacija (cikličko pomeranje).
  · Attention (kvantno) = SWAP-test(Q_h, K_b) → |⟨·|·⟩|² (bez softmax-a).
  · Multi-head mix = (1/H) Σ_h p_h.
  · Kvantni FFN = StatePreparation(√p_layer) → PQC (Ry(π·p·dim) + ring-CNOT)
    → |Q⁽ˡ⁺¹⁾⟩ = √|ψ|² (L2-renorm).
Izlaz: p iz poslednjeg sloja → bias_39 → TOP-7 = NEXT.

Tehnike:
SWAP test za kvantnu meru sličnosti.
Multi-head via roll(amp) — deterministička baza po glavi.
LCU-slična agregacija p-stanja po blokovima (linearna mešavina).
Kvantni FFN kao PQC inicijalizovan stanjem sloja + ring-CNOT + Ry.
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, H, B, L).

Prednosti:
Pravi multi-head + L slojeva (nadogradnja nad Q10).
Čisto kvantno: attention iz interferencije, FFN kao PQC, bez klasičnog ML-a.
Svi parametri deterministički izvedeni iz celog CSV-a.

Nedostaci:
Per sloj/glava koristi 2·nq + 1 qubit-a (SWAP test) — budžet drži nq ≤ 6, H ≤ 3.
Mešavina p-stanja preko glava/blokova je linearna (nije LCU-unitary).
mod-39 readout meša stanja (dim 2^nq ≠ 39).
Mera cos(bias, freq_csv) je pristrana ka reprodukciji marginale.
"""
