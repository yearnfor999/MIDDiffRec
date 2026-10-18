"""
Repeated-generation Stability Evaluation for MIDDiffRec vs Single-condition.

Usage — run from within a model directory:

  # MIDDiffRec (full)
  cd ~/19/InDiRec-main-all
  python ../stability_eval.py \
      --data_name Beauty \
      --checkpoint InDiRec-Beauty-91.pt \
      --model_name MIDDiffRec \
      --timesteps 50 --intent_num 32 --alpha 0.7 --c 5 --w 1 --p 0.2 \
      --num_seeds 10 --num_samples 1000 \
      --output_json ../results_MIDDiffRec.json

  # Single-condition (x_t + m_u + e_t)
  cd ~/19/InDiRec-main-woLS
  python ../stability_eval.py \
      --data_name Beauty \
      --checkpoint output/InDiRec-Beauty-306.pt \
      --model_name Single-condition \
      --ablate_long_term --ablate_short_term \
      --timesteps 50 --intent_num 32 --alpha 0.7 --c 5 --w 1 --p 0.2 \
      --num_seeds 10 --num_samples 1000 \
      --output_json ../results_Single.json
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from models import InDiRec, KMeans
from datasets import get_seqs_and_matrixes, DatasetForInDiRec, D
from trainer import InDiRecTrainer
from utils import set_seed


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", default="./datasets/", type=str)
    p.add_argument("--data_name", default="Beauty", type=str)
    p.add_argument("--max_seq_length", type=int, default=50)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--seed", default=2022, type=int)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--without_segment", action="store_true")
    p.add_argument("--noise_ratio", type=float, default=0.0)

    # model architecture
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_hidden_layers", type=int, default=2)
    p.add_argument("--num_attention_heads", type=int, default=2)
    p.add_argument("--hidden_act", type=str, default="gelu")
    p.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    p.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    p.add_argument("--initializer_range", type=float, default=0.02)

    # diffusion
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--beta_start", type=float, default=0.0001)
    p.add_argument("--beta_sche", nargs="?", default="exp")
    p.add_argument("--diffuser_type", type=str, default="mlp1")
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--p", type=float, default=0.2)
    p.add_argument("--use_ddim", action="store_true")
    p.add_argument("--ddim_steps", type=int, default=20)

    # intent / BSARec
    p.add_argument("--intent_num", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--c", type=int, default=5)

    # ablation flags (for single-condition version)
    p.add_argument("--ablate_long_term", action="store_true")
    p.add_argument("--ablate_short_term", action="store_true")

    # trainer dummy args (not used in eval, but required by Trainer.__init__)
    p.add_argument("--rec_weight", type=float, default=1.0)
    p.add_argument("--diff_weight", type=float, default=1.0)
    p.add_argument("--diffusion_rec_weight", type=float, default=0.01)
    p.add_argument("--user_weight", type=float, default=0.05)
    p.add_argument("--short_term_weight", type=float, default=0.01)
    p.add_argument("--cl_weight", type=float, default=0.2)
    p.add_argument("--tau", type=int, default=-1)
    p.add_argument("--sim", type=str, default="dot")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--cl_use_reverse_process", action="store_true")
    p.add_argument("--model_idx", default=0, type=int)
    p.add_argument("--model_name", default="InDiRec", type=str)
    p.add_argument("--use_single_step_cfg", action="store_true")
    p.add_argument("--single_step_cfg_w", type=float, default=2.0)

    # stability eval specific
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--num_seeds", type=int, default=10,
                   help="Number of random seeds for repeated sampling")
    p.add_argument("--num_samples", type=int, default=1000,
                   help="Max number of test users to evaluate")
    p.add_argument("--output_json", type=str, default="stability_results.json")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Rebuild KMeans + user intent distribution (same as trainer.iteration)
# ---------------------------------------------------------------------------

def rebuild_intent_state(trainer, cluster_dataloader, device):
    print("Rebuilding KMeans clustering...")
    trainer.model.eval()

    kmeans_data = []
    seq_ids = []
    counter = 0

    with torch.no_grad():
        for batch in tqdm(cluster_dataloader, desc="  Extracting features"):
            batch = tuple(t.to(device) for t in batch)
            _, subseq, _, _, _ = batch
            feat = trainer.seq_model(subseq)[:, -1, :]
            kmeans_data.append(feat.detach().cpu().numpy())
            B = subseq.size(0)
            seq_ids.extend(range(counter, counter + B))
            counter += B

    kmeans_data = np.concatenate(kmeans_data, axis=0).astype("float32")
    seq_ids = np.array(seq_ids)

    for i, clusters in enumerate(trainer.clusters_t):
        for j, cluster in enumerate(clusters):
            cluster.train(kmeans_data, seq_ids)
            trainer.clusters_t[i][j] = cluster

    trainer.build_user_intent_distribution(cluster_dataloader)
    print("Intent state rebuilt.")


# ---------------------------------------------------------------------------
# Single-seed reverse diffusion (DDPM, full stochasticity)
# ---------------------------------------------------------------------------

def sample_one_seed(model, input_seq, user_ids, m_u, seed, args):
    """Run one complete reverse diffusion with a fixed seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.no_grad():
        x0_hat = model.sample_from_reverse_process_enhanced(
            input_seq, user_ids, m_u,
            use_ddim=False,       # DDPM: randomness at every step
            ddim_steps=args.ddim_steps,
        )
    return x0_hat.detach()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(views, h_seq, answers, item_emb):
    """
    views    : [S, B, D]  – S generated views per user
    h_seq    : [B, D]     – sequence encoder output (deterministic anchor)
    answers  : [B]        – ground-truth next item id
    item_emb : [V, D]     – item embedding matrix
    """
    S, B, _ = views.shape

    # ---- scores and ranks ------------------------------------------------
    scores = torch.matmul(views, item_emb.T)              # [S, B, V]

    # gather target item score: answers [B] → expand to [S, B, 1] → squeeze → [S, B]
    answers_exp = answers.view(1, B, 1).expand(S, -1, -1)          # [S, B, 1]
    target_scores = scores.gather(dim=2, index=answers_exp).squeeze(-1)  # [S, B]

    std_score = target_scores.std(dim=0)                  # [B]

    # rank = 1 + #items scoring strictly higher than the target
    ranks = 1.0 + (scores > target_scores.unsqueeze(-1)).float().sum(dim=-1)  # [S, B]
    std_rank = ranks.std(dim=0)                           # [B]

    # ---- semantic alignment with h_seq -----------------------------------
    h_norm = F.normalize(h_seq, dim=-1)                   # [B, D]
    v_norm = F.normalize(views, dim=-1)                   # [S, B, D]
    sims = (v_norm * h_norm.unsqueeze(0)).sum(dim=-1)     # [S, B]
    sim_mean = sims.mean(dim=0)                           # [B]
    sim_std  = sims.std(dim=0)                            # [B]

    # ---- pairwise cosine distance (vectorised) ----------------------------
    cos_mat = torch.einsum("ibd,jbd->ijb", v_norm, v_norm)   # [S, S, B]
    dist_mat = 1.0 - cos_mat                                   # [S, S, B]
    triu = torch.triu(torch.ones(S, S, device=views.device), diagonal=1).bool()
    triu_b = triu.unsqueeze(-1).expand(-1, -1, B)
    pair_dist = dist_mat[triu_b].view(-1, B).mean(dim=0)      # [B]

    # ---- Jaccard@20 (CPU set ops) -----------------------------------------
    top20 = scores.topk(k=20, dim=-1).indices.cpu()  # [S, B, 20]
    n_pairs = S * (S - 1) // 2
    jaccard = []
    for b in range(B):
        jac = 0.0
        for i in range(S):
            for j in range(i + 1, S):
                si = set(top20[i, b].tolist())
                sj = set(top20[j, b].tolist())
                inter = len(si & sj)
                union = len(si | sj)
                jac += inter / union if union > 0 else 0.0
        jaccard.append(jac / n_pairs)

    return {
        "std_rank":  std_rank.cpu().tolist(),
        "std_score": std_score.cpu().tolist(),
        "sim_mean":  sim_mean.cpu().tolist(),
        "sim_std":   sim_std.cpu().tolist(),
        "jaccard":   jaccard,
        "pair_dist": pair_dist.cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_stability_eval(model, trainer, test_dataloader, device, args):
    S = args.num_seeds
    item_emb = model.item_embeddings.weight.detach()  # [V, D]

    agg = {k: [] for k in
           ["std_rank", "std_score", "sim_mean", "sim_std", "jaccard", "pair_dist"]}

    sample_count = 0

    for batch in tqdm(test_dataloader, desc=f"Stability ({args.model_name}, S={S})"):
        if sample_count >= args.num_samples:
            break

        batch = tuple(t.to(device) for t in batch)
        user_ids, input_seq, _, target_pos, answers = batch
        B = input_seq.size(0)

        with torch.no_grad():
            # Deterministic conditions — computed once per batch
            h_seq = model.seq_model(input_seq)[:, -1, :]          # [B, D]

            user_dist = trainer.user_intent_hist[user_ids]         # [B, K]
            m_u = model.process_intent_distribution(user_dist)     # [B, D]

        # S repeated samplings with different seeds
        views = torch.stack(
            [sample_one_seed(model, input_seq, user_ids, m_u,
                             seed=args.seed + s, args=args)
             for s in range(S)],
            dim=0,
        )  # [S, B, D]

        metrics = compute_metrics(views, h_seq.detach(), answers, item_emb)
        for k in agg:
            agg[k].extend(metrics[k])

        sample_count += B

    results = {
        "model":       args.model_name,
        "dataset":     args.data_name,
        "num_seeds":   S,
        "num_samples": sample_count,
        "StdRank":     float(np.mean(agg["std_rank"])),
        "StdScore":    float(np.mean(agg["std_score"])),
        "SimMean":     float(np.mean(agg["sim_mean"])),
        "SimStd":      float(np.mean(agg["sim_std"])),
        "Jaccard@20":  float(np.mean(agg["jaccard"])),
        "PairDist":    float(np.mean(agg["pair_dist"])),
    }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.data_file      = args.data_dir + args.data_name + ".txt"
    args.segmented_file = args.data_dir + args.data_name + "_s.txt"
    if not os.path.exists(args.segmented_file):
        D(args.data_file, args.segmented_file, args.max_seq_length)

    # ---- data ---------------------------------------------------------------
    train_user_ids, train_seq = get_seqs_and_matrixes("training", args.segmented_file)
    eval_user_ids, user_seq, max_item, _, test_rating_matrix = \
        get_seqs_and_matrixes("rating", args.data_file)

    all_uids     = set(train_user_ids + eval_user_ids)
    sorted_uids  = sorted(all_uids)
    uid_map      = {uid: idx for idx, uid in enumerate(sorted_uids)}
    args.num_users   = len(sorted_uids)
    args.user_id_map = uid_map
    args.item_size   = max_item + 2
    args.mask_id     = max_item + 1
    args.train_matrix = test_rating_matrix
    args.log_file    = args.output_json.replace(".json", ".log")

    train_indices = [uid_map[u] for u in train_user_ids]
    eval_indices  = [uid_map[u] for u in eval_user_ids]

    cluster_dataset = DatasetForInDiRec(args, train_seq,
                                        user_ids=train_indices, data_type="train")
    cluster_loader  = DataLoader(cluster_dataset,
                                 sampler=SequentialSampler(cluster_dataset),
                                 batch_size=args.batch_size, drop_last=False)

    test_dataset = DatasetForInDiRec(args, user_seq,
                                     user_ids=eval_indices, data_type="test")
    test_loader  = DataLoader(test_dataset,
                              sampler=SequentialSampler(test_dataset),
                              batch_size=args.batch_size, drop_last=False)

    # ---- model --------------------------------------------------------------
    model = InDiRec(device, args=args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # ---- trainer (needed only for KMeans infrastructure) --------------------
    trainer = InDiRecTrainer(model, [], cluster_loader, [], [], device, args)

    # ---- rebuild intent state -----------------------------------------------
    rebuild_intent_state(trainer, cluster_loader, device)

    # ---- evaluate -----------------------------------------------------------
    results = run_stability_eval(model, trainer, test_loader, device, args)

    print("\n=== Stability Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()
