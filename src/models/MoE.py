import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import entmax_bisect
from torch_geometric.nn import global_mean_pool

from models.expert import Experts
from models.gnn_models import GINEncoderWithEdgeWeight


class MoE(nn.Module):
    """
    Mixture-of-Experts wrapper:
      - owns a shared Experts module (K experts inside);
      - owns a gate over the RAW input graph;
      - aggregates per-expert losses using (avg) gate weights.
    """
    def __init__(self, config, dataset_info):
        super().__init__()
        self.verbose         = config['experiment']['debug']['verbose']
        self.num_experts     = config['model']['num_experts']
        self.aggregation     = config['model']['aggregation']     # expect 'weighted_mean'
        self.train_after     = config['gate']['train_after']
        self.weight_ce       = config['model']['weight_ce']
        self.weight_reg      = config['model']['weight_reg']
        self.weight_la      = config['model']['weight_la']
        self.weight_ea      = config['model']['weight_ea']
        self.weight_str      = config['model']['weight_str']
        self.weight_div      = config['model']['weight_div']
        self.weight_load     = config['model']['weight_load']
        self.topk_train_k    = int(config['gate'].get('topk_train_k', 2))    # 0 => disabled
        self.topk_after      = int(config['gate'].get('topk_after', self.train_after + 1))

        # Shared expert block (contains all experts)
        self.shared = Experts(config, dataset_info)

        # Gating network on raw graph
        self.num_features   = dataset_info['num_features']
        self.num_classes    = dataset_info['num_classes']
        self.metric         = dataset_info['metric']

        gate_hidden    = config['gate']['hidden_dim']
        gate_depth     = config['gate']['depth']
        dropout        = config['model']['dropout']
        self.gate_enc  = GINEncoderWithEdgeWeight(self.num_features, gate_hidden, gate_depth, dropout, train_eps=True)
        self.gate_mlp  = nn.Sequential(nn.Linear(gate_hidden, gate_hidden),
                                       nn.ReLU(),
                                       nn.Linear(gate_hidden, self.num_experts))
        self.entmax_alpha = float(config['gate']['entmax_alpha'])

        self.current_epoch = 0

        if self.verbose:
            print(f"[MoE] K={self.num_experts}, aggregation={self.aggregation}, gate=raw-graph")

    # ------------- Public API -------------
    def set_epoch(self, epoch: int):
        """Forward epoch to experts and update gate schedule (warm-up)."""
        self.current_epoch = int(epoch)
        self.shared.set_epoch(epoch)

    def forward(self, data):
        # Run experts
        shared_out = self.shared(data, data.y)  # includes per-expert lists + diversity

        # Gate on RAW graph (no expert info)
        gate_scores = self._gate_logits_raw(data)  # (B, K)
        # NEW: compute per-sample expert uncertainty u \in [0,1]
        # u = mean_k H(p_k) / log(C), where p_k are per-expert class probs
        # expert_logits = shared_out['expert_logits']        # (B,K,C)
        # B, K, C = expert_logits.shape
        # with torch.no_grad():
        #     per_expert_log_probs = expert_logits.log_softmax(dim=-1)        # (B,K,C)
        #     per_expert_probs     = per_expert_log_probs.exp()
        #     per_expert_entropy   = -(per_expert_probs * per_expert_log_probs).sum(dim=-1)   # (B,K)
        #     u = per_expert_entropy.mean(dim=1) / (torch.log(torch.tensor(float(C), device=expert_logits.device)))  # (B,)


        if self.current_epoch < self.train_after:
            # Freeze gate to uniform early on
            gate_probs = torch.full_like(gate_scores, 1.0 / self.num_experts)
        else:
            if self.entmax_alpha > 1:
                gate_probs = entmax_bisect(gate_scores, alpha=self.entmax_alpha, dim=-1)
            else:
                gate_probs = F.softmax(gate_scores, dim=-1)
            # # --- in MoE.forward, right after gate_probs is computed (and blended) ---
            # if (self.current_epoch >= self.topk_after) and (self.topk_train_k and self.topk_train_k < self.num_experts):
            #     # hard per-sample top-k mask
            #     topk_vals, topk_idx = gate_probs.topk(self.topk_train_k, dim=1)       # (B, k)
            #     mask = torch.zeros_like(gate_probs)                                    # (B, K)
            #     mask.scatter_(1, topk_idx, 1.0)
            #     gate_probs = gate_probs * mask
            #     denom = gate_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            #     gate_probs = gate_probs / denom

        if self.verbose:
            with torch.no_grad():
                print("[MoE] gate (first row):", gate_probs[0].tolist())

        # Aggregate logits and losses
        out = self._aggregate(shared_out, gate_probs, targets=data.y)
        return out

    # ------------- Internals -------------
    def _gate_logits_raw(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.gate_enc(x, edge_index, batch=batch)
        h = global_mean_pool(h, batch)
        return self.gate_mlp(h)  # (B, K)

    def _aggregate(self, shared_out, gate_probs, targets):
        """
        Gate-weighted aggregation:
          - logits: per-sample gate-weighted sum of expert logits
          - CE:    per-expert, per-sample CE weighted by that sample's gate prob for that expert
          - other losses: gate-averaged across batch
        """
        expert_logits = shared_out['expert_logits']       # (B, K, C)
        B, K, C = expert_logits.shape

        # logits aggregation
        weighted = expert_logits * gate_probs.unsqueeze(-1)           # (B, K, C)
        agg_logits = weighted.sum(dim=1)                              # (B, C)

        # CE aggregated per expert, per sample
        ce = self._gate_weighted_ce(expert_logits.transpose(0, 1), targets, gate_probs.transpose(0, 1), self.metric, self.num_classes)

        # Other per-expert losses (scalars) — combine via avg gate weights over batch
        avg_gate = gate_probs.mean(dim=0)  # (K,)
        reg = (avg_gate * shared_out['loss_reg_list']).sum()
        la = (avg_gate * shared_out['loss_la_list']).sum()
        ea = (avg_gate * shared_out['loss_ea_list']).sum()
        strl = (avg_gate * shared_out['loss_str_list']).sum()

        # Diversity (already scalar); Load-balance on gate (optional)
        div = shared_out['loss_div']
        load = self._load_balance(gate_probs)

        total = (self.weight_ce * ce +
                 self.weight_reg * reg +
                 self.weight_la * la +
                 self.weight_ea * ea +
                 self.weight_str * strl +
                 self.weight_div * div +
                 self.weight_load * load)

        return {
            'logits': agg_logits,
            'loss_total': total,
            'loss_ce': ce * self.weight_ce,
            'loss_reg': reg * self.weight_reg,
            'loss_la': la * self.weight_la,
            'loss_ea': ea * self.weight_ea,
            'loss_str': strl * self.weight_str,
            'loss_div': div * self.weight_div,
            'loss_load': load * self.weight_load,
            'gate_weights': gate_probs,              # (B, K)
            'rho': shared_out['rho'],
            'expert_logits': expert_logits,         # (B, K, C)
            'node_masks': shared_out['node_masks'],
            'edge_masks': shared_out['edge_masks'],
            'feat_masks': shared_out['feat_masks'],
        }

    @staticmethod
    def _gate_weighted_ce(stacked_logits, targets, gate_weights, metric, num_classes):
        """
        stacked_logits: (K, B, C)
        gate_weights:   (K, B)
        """
        K, B, _ = stacked_logits.shape
        if metric == 'MAE' and num_classes == 1:
            # expect C == 1
            pred = stacked_logits[..., 0]       # (K, B)
            tgt  = targets.view(B).to(pred.dtype)  # (B,)

            # elementwise absolute error, no reduction
            per_exp_sample = F.l1_loss(pred, tgt.unsqueeze(0).expand_as(pred), reduction='none')  # (K,B)

            # gate-weighted mean across experts and batch
            loss = (gate_weights * per_exp_sample).mean()
        elif metric == 'Accuracy' and num_classes == 1:
        # one logit per sample
            logits = stacked_logits[..., 0]      # (K,B)
            tgt    = targets.view(B).to(logits.dtype)  # (B,)
            # BCEWithLogits per-sample per-expert
            per_exp_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, tgt.unsqueeze(0).expand(K, B), reduction='none'
            )  # (K,B)
            loss = (gate_weights * per_exp_sample).mean()
        else:
            loss = stacked_logits[0, :, 0].new_tensor(0.0)
            for k in range(K):
                ce_per_sample = F.cross_entropy(stacked_logits[k], targets, reduction='none')  # (B,)
                loss += (gate_weights[k] * ce_per_sample).mean()
        return loss

    @staticmethod
    def _load_balance(gate_probs, lam=0.2, T=0.2, eps=1e-8):
        # gate_probs: (B,K)
        B, K = gate_probs.shape
        uniform = gate_probs.new_full((K,), 1.0 / K)

        # 1) Balanced usage across batch (nonnegative, 0 at uniform marginal)
        p_bar = gate_probs.mean(0)
        L_bal = F.kl_div((p_bar + eps).log(), uniform, reduction='batchmean')

        # 2) Per-sample sharpness (penalize uniform rows)
        H_rows = -(gate_probs * (gate_probs + eps).log()).sum(dim=1).mean()

        # 3) Spread winners across experts (soft top-1 histogram)
        q = F.softmax((gate_probs + eps).log() / T, dim=-1)  # sharpened rows
        counts = q.mean(dim=0)
        L_top1 = F.kl_div((counts + eps).log(), uniform, reduction='batchmean')

        return L_bal + lam * H_rows + 0.2 * L_top1
