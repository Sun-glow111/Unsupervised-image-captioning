import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 配置
# ============================================================

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据相关
    vocab_size: int = 5000
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    max_text_len: int = 20
    image_feat_dim: int = 512

    # 模型维度
    hidden_dim: int = 512
    shared_dim: int = 256
    private_dim: int = 256
    plan_slots: int = 6
    plan_dim: int = 256
    num_decoder_layers: int = 4
    num_decoder_heads: int = 8
    dropout: float = 0.1

    # 训练相关
    batch_size: int = 32
    lr: float = 3e-4
    num_epochs_stage1: int = 3   # 文本侧: text -> plan -> text
    num_epochs_stage2: int = 3   # 图像侧: image -> plan -> text
    lambda_disent: float = 0.1
    lambda_plan_align: float = 0.5
    lambda_plan_lm: float = 0.2
    grad_clip: float = 1.0


# ============================================================
# 2. 一个最小可跑的数据集（随机数据占位）
#    你以后把它替换成真实图像特征和真实 caption token 即可
# ============================================================

class DummyImageDataset(Dataset):
    def __init__(self, n: int, image_feat_dim: int):
        self.n = n
        self.image_feat_dim = image_feat_dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # 用随机图像特征占位；真实情况应替换为预提取视觉特征
        feat = torch.randn(self.image_feat_dim)
        return {"image_feat": feat}


class DummyTextDataset(Dataset):
    def __init__(self, n: int, vocab_size: int, max_len: int, bos_id: int, eos_id: int):
        self.n = n
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        length = random.randint(6, self.max_len)
        middle = torch.randint(3, self.vocab_size, (length - 2,))
        tokens = torch.cat([
            torch.tensor([self.bos_id]),
            middle,
            torch.tensor([self.eos_id])
        ], dim=0)
        return {"input_ids": tokens}


def pad_text_batch(batch: List[Dict], pad_id: int):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - ids.shape[0]
        padded = F.pad(ids, (0, pad_len), value=pad_id)
        mask = torch.ones_like(padded)
        if pad_len > 0:
            mask[-pad_len:] = 0
        input_ids.append(padded)
        attention_mask.append(mask)
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
    }


def collate_image(batch: List[Dict]):
    return {"image_feat": torch.stack([x["image_feat"] for x in batch], dim=0)}


# ============================================================
# 3. 编码器（简化版）
#    真实实现里可以换成冻结的 CLIP/DINOv2/BERT 等
# ============================================================

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, pad_id: int, dropout: float):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(256, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = self.norm(x)

        # 句级表示：对非 pad token 做 masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        sent = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return {
            "token_feats": x,     # [B, T, H]
            "sent_feat": sent,    # [B, H]
        }


class SimpleImageEncoder(nn.Module):
    def __init__(self, image_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(image_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, image_feat: torch.Tensor):
        # 简化主干：只有全局图像特征
        x = self.proj(image_feat)
        return {
            "global_feat": x,  # [B, H]
        }


# ============================================================
# 4. shared/private 分解
# ============================================================

class SharedPrivateProjector(nn.Module):
    def __init__(self, hidden_dim: int, shared_dim: int, private_dim: int):
        super().__init__()
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )
        self.private_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, private_dim),
            nn.LayerNorm(private_dim),
        )

    def forward(self, feat: torch.Tensor):
        s = self.shared_head(feat)
        p = self.private_head(feat)
        return s, p


def disentanglement_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    # 目标：shared 和 private 尽量少相关
    shared = F.normalize(shared, dim=-1)
    private = F.normalize(private, dim=-1)
    corr = (shared * private).sum(dim=-1).abs().mean()
    return corr


# ============================================================
# 5. semantic plan 生成器
#    文本侧: token_feats -> slots
#    图像侧: global_feat -> slots
# ============================================================

class SlotPlannerFromText(nn.Module):
    def __init__(self, hidden_dim: int, plan_slots: int, plan_dim: int, dropout: float):
        super().__init__()
        self.plan_slots = plan_slots
        self.query = nn.Parameter(torch.randn(plan_slots, plan_dim))
        self.key_proj = nn.Linear(hidden_dim, plan_dim)
        self.val_proj = nn.Linear(hidden_dim, plan_dim)
        self.out = nn.Sequential(
            nn.Linear(plan_dim, plan_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(plan_dim),
        )

    def forward(self, token_feats: torch.Tensor, attention_mask: torch.Tensor):
        # token_feats: [B, T, H]
        B, T, _ = token_feats.shape
        K = self.key_proj(token_feats)      # [B, T, D]
        V = self.val_proj(token_feats)      # [B, T, D]
        Q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        attn = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.shape[-1])  # [B, M, T]
        attn = attn.masked_fill((attention_mask == 0).unsqueeze(1), -1e9)
        attn = F.softmax(attn, dim=-1)
        slots = torch.matmul(attn, V)       # [B, M, D]
        slots = self.out(slots)
        return slots


class SlotPlannerFromImage(nn.Module):
    def __init__(self, shared_dim: int, plan_slots: int, plan_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, plan_slots * plan_dim),
        )
        self.plan_slots = plan_slots
        self.plan_dim = plan_dim
        self.norm = nn.LayerNorm(plan_dim)

    def forward(self, shared_feat: torch.Tensor):
        B = shared_feat.shape[0]
        z = self.net(shared_feat).view(B, self.plan_slots, self.plan_dim)
        z = self.norm(z)
        return z


def add_plan_noise(plan: torch.Tensor, dropout_p: float = 0.1, noise_std: float = 0.01):
    if dropout_p > 0:
        keep = (torch.rand(plan.shape[:2], device=plan.device) > dropout_p).float().unsqueeze(-1)
        plan = plan * keep
    if noise_std > 0:
        plan = plan + torch.randn_like(plan) * noise_std
    return plan


# ============================================================
# 6. 文本解码器: plan -> text
# ============================================================

class PlanConditionedDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, plan_dim: int, num_layers: int, num_heads: int, pad_id: int, dropout: float):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(256, hidden_dim)
        self.plan_proj = nn.Linear(plan_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def causal_mask(self, T: int, device: torch.device):
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, plan: torch.Tensor, input_ids: torch.Tensor):
        # plan: [B, M, Dp]
        # input_ids: [B, T]
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        tgt = self.token_emb(input_ids) + self.pos_emb(pos)
        memory = self.plan_proj(plan)  # [B, M, H]
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=self.causal_mask(T, input_ids.device))
        logits = self.lm_head(out)
        return logits

    @torch.no_grad()
    def generate(self, plan: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 20):
        B = plan.shape[0]
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=plan.device)
        for _ in range(max_len - 1):
            logits = self.forward(plan, ys)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return ys


def language_modeling_loss(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int):
    # teacher forcing: 输入通常是 y[:, :-1]，目标是 y[:, 1:]
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=pad_id,
    )


# ============================================================
# 7. 主模型
# ============================================================

class UnpairedCaptionBackbone(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.text_encoder = SimpleTextEncoder(cfg.vocab_size, cfg.hidden_dim, cfg.pad_id, cfg.dropout)
        self.image_encoder = SimpleImageEncoder(cfg.image_feat_dim, cfg.hidden_dim)

        self.text_proj = SharedPrivateProjector(cfg.hidden_dim, cfg.shared_dim, cfg.private_dim)
        self.image_proj = SharedPrivateProjector(cfg.hidden_dim, cfg.shared_dim, cfg.private_dim)

        self.text_planner = SlotPlannerFromText(cfg.hidden_dim, cfg.plan_slots, cfg.plan_dim, cfg.dropout)
        self.image_planner = SlotPlannerFromImage(cfg.shared_dim, cfg.plan_slots, cfg.plan_dim, cfg.dropout)

        self.decoder = PlanConditionedDecoder(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.hidden_dim,
            plan_dim=cfg.plan_dim,
            num_layers=cfg.num_decoder_layers,
            num_heads=cfg.num_decoder_heads,
            pad_id=cfg.pad_id,
            dropout=cfg.dropout,
        )

    # -------------------------
    # Stage 1: text -> plan -> text
    # -------------------------
    def forward_text_stage(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        enc = self.text_encoder(input_ids, attention_mask)
        s, p = self.text_proj(enc["sent_feat"])
        plan = self.text_planner(enc["token_feats"], attention_mask)
        noisy_plan = add_plan_noise(plan, dropout_p=0.1, noise_std=0.01)

        dec_in = input_ids[:, :-1]
        dec_tgt = input_ids[:, 1:]
        logits = self.decoder(noisy_plan, dec_in)
        txt_loss = language_modeling_loss(logits, dec_tgt, self.cfg.pad_id)
        dis_loss = disentanglement_loss(s, p)

        total = txt_loss + self.cfg.lambda_disent * dis_loss
        return {
            "loss": total,
            "txt_loss": txt_loss.detach(),
            "dis_loss": dis_loss.detach(),
            "text_shared": s,
            "text_private": p,
            "text_plan": plan.detach(),
        }

    # -------------------------
    # Stage 2: image -> plan -> text
    # 这里为了“能跑通”，我们用同 batch 的文本 plan 作为分布锚点
    # 真正 fully unpaired 时，可替换成 memory bank / distribution matching
    # -------------------------
    def forward_joint_stage(self, image_batch: Dict[str, torch.Tensor], text_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 文本侧：提供可解码 plan 目标
        text_ids = text_batch["input_ids"]
        text_mask = text_batch["attention_mask"]
        t_enc = self.text_encoder(text_ids, text_mask)
        t_shared, t_private = self.text_proj(t_enc["sent_feat"])
        text_plan = self.text_planner(t_enc["token_feats"], text_mask)

        # 图像侧
        image_feat = image_batch["image_feat"]
        i_enc = self.image_encoder(image_feat)
        i_shared, i_private = self.image_proj(i_enc["global_feat"])
        image_plan = self.image_planner(i_shared)

        # 1) 图像/文本分解损失
        dis_loss = disentanglement_loss(t_shared, t_private) + disentanglement_loss(i_shared, i_private)

        # 2) 最小可运行版的 plan 对齐
        #    这里不是样本级真配对，只是把两边 batch plan 的均值分布拉近
        plan_align_loss = F.mse_loss(image_plan.mean(dim=0), text_plan.mean(dim=0))

        # 3) 为了让 image_plan 真能解码，拿同 batch 文本当作语言目标做弱监督训练
        #    这是简化版主干，不是最终 fully unpaired 论文版
        dec_in = text_ids[:, :-1]
        dec_tgt = text_ids[:, 1:]
        logits = self.decoder(image_plan, dec_in)
        image_lm_loss = language_modeling_loss(logits, dec_tgt, self.cfg.pad_id)

        total = (
            self.cfg.lambda_disent * dis_loss
            + self.cfg.lambda_plan_align * plan_align_loss
            + self.cfg.lambda_plan_lm * image_lm_loss
        )

        return {
            "loss": total,
            "dis_loss": dis_loss.detach(),
            "plan_align_loss": plan_align_loss.detach(),
            "image_lm_loss": image_lm_loss.detach(),
            "image_shared": i_shared.detach(),
            "text_shared": t_shared.detach(),
            "image_plan": image_plan.detach(),
            "text_plan": text_plan.detach(),
        }

    @torch.no_grad()
    def generate_from_image(self, image_feat: torch.Tensor):
        if image_feat.dim() == 1:
            image_feat = image_feat.unsqueeze(0)
        enc = self.image_encoder(image_feat)
        i_shared, _ = self.image_proj(enc["global_feat"])
        image_plan = self.image_planner(i_shared)
        return self.decoder.generate(image_plan, self.cfg.bos_id, self.cfg.eos_id, self.cfg.max_text_len)


# ============================================================
# 8. 训练与验证
# ============================================================

def move_to_device(batch: Dict[str, torch.Tensor], device: str):
    return {k: v.to(device) for k, v in batch.items()}


def train_stage1_text(model: UnpairedCaptionBackbone, loader: DataLoader, optimizer, cfg: Config):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_to_device(batch, cfg.device)
        out = model.forward_text_stage(batch)
        optimizer.zero_grad()
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += out["loss"].item()
    return total_loss / max(len(loader), 1)


def train_stage2_joint(model: UnpairedCaptionBackbone, image_loader: DataLoader, text_loader: DataLoader, optimizer, cfg: Config):
    model.train()
    total_loss = 0.0
    text_iter = iter(text_loader)
    for image_batch in image_loader:
        try:
            text_batch = next(text_iter)
        except StopIteration:
            text_iter = iter(text_loader)
            text_batch = next(text_iter)

        image_batch = move_to_device(image_batch, cfg.device)
        text_batch = move_to_device(text_batch, cfg.device)
        out = model.forward_joint_stage(image_batch, text_batch)

        optimizer.zero_grad()
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += out["loss"].item()
    return total_loss / max(len(image_loader), 1)


@torch.no_grad()
def demo_generate(model: UnpairedCaptionBackbone, image_loader: DataLoader, cfg: Config):
    model.eval()
    batch = next(iter(image_loader))
    image_feat = batch["image_feat"].to(cfg.device)
    token_ids = model.generate_from_image(image_feat[:2])
    return token_ids.cpu()


# ============================================================
# 9. 主程序
# ============================================================

def main():
    cfg = Config()
    print("Using device:", cfg.device)

    # 独立图片集、独立文本集（dummy 占位）
    image_ds = DummyImageDataset(n=256, image_feat_dim=cfg.image_feat_dim)
    text_ds = DummyTextDataset(
        n=256,
        vocab_size=cfg.vocab_size,
        max_len=cfg.max_text_len,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
    )

    image_loader = DataLoader(image_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_image)
    text_loader = DataLoader(text_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: pad_text_batch(b, cfg.pad_id))

    model = UnpairedCaptionBackbone(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Stage 1: 先学 text -> plan -> text
    print("\n===== Stage 1: train text-plan decoder =====")
    for epoch in range(cfg.num_epochs_stage1):
        loss = train_stage1_text(model, text_loader, optimizer, cfg)
        print(f"[Stage1][Epoch {epoch+1}] loss={loss:.4f}")

    # Stage 2: 再学 image -> plan，并接同一个 decoder
    print("\n===== Stage 2: train image-plan backbone =====")
    for epoch in range(cfg.num_epochs_stage2):
        loss = train_stage2_joint(model, image_loader, text_loader, optimizer, cfg)
        print(f"[Stage2][Epoch {epoch+1}] loss={loss:.4f}")
        sample = demo_generate(model, image_loader, cfg)
        print("Generated token ids sample:")
        print(sample)

    print("\nDone.")


if __name__ == "__main__":
    main()
