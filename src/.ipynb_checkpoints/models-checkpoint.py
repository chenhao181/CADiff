# a variant of GRU4REC: change from session-based next-item recommendation to sequential recommendation
import math
import torch
from data_loader import Dataset
from step_sample import create_named_schedule_sampler
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_hidden, device, n_head=4, n_layers=4):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(n_hidden)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=n_hidden, nhead=n_head, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp_output = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden))
        self.mlp_user = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden))
        self.to(device)
        self.device = device

    def forward(self, xs, length):
        shape = xs.shape
        xs = xs.view(-1, shape[-2], shape[-1])

        xs = self.pos_encoder(xs)
        # 构造 mask
        # True 表示屏蔽
        mask = torch.zeros(xs.shape[1], xs.shape[1], dtype=torch.bool).to(xs.device)
        # mask[0, -1] = True
        # 1. 其他 item 看不到第一个 item
        mask[1:, 0] = True
        # for i in range(1, xs.shape[1]):
        #     mask[i, i + 1:] = False
        xs = self.transformer(xs, mask=mask)
        # xs = self.mlp_output(xs[torch.arange(xs.size(0)).to(self.device), length - 1])
        inter_embedding = xs[:, -1]
        user_embedding = xs[:, 0]
        # xs = xs[:, -1]
        return inter_embedding.view(*shape[:-2], shape[-1]), user_embedding.view(*shape[:-2], shape[-1])

class TimeAwareAttentionFusion(nn.Module):
    def __init__(self, d_model, time_embed_dim=64):
        super().__init__()
        # self.time_embed = nn.Linear(1, time_embed_dim)
        # self.attn_fc = nn.Linear(d_model * 2 + time_embed_dim, 2)
        self.attn_fc = nn.Linear(d_model * 2 , 2)
        # self.attn_fc = nn.Sequential(
        #     nn.Linear(d_model * 2, d_model),
        #     nn.GELU(),
        #     nn.Linear(d_model, 2)
        # )
        self.gelu = nn.GELU()

    def forward(self, c, s, t, T):
        # tau = (t / (T - 1)).unsqueeze(-1)  # (B,1)
        # t_emb = self.gelu(self.time_embed(tau))
        # cat_feat = torch.cat([c, s, t_emb], dim=-1)
        cat_feat = torch.cat([c, s], dim=-1)
        logits = self.attn_fc(cat_feat)
        attn = F.softmax(logits , dim=-1)
        fused = attn[:, 0:1] * c + attn[:, 1:2] * s
        return attn, fused


class GCDR(torch.nn.Module):  # Conditional Diffusion Recommender Model
    def __init__(self, dataset: Dataset, device, args):
        super(GCDR, self).__init__()
        self.dataset = dataset
        self.device = device
        self.n = 10
        # Not in the paper
        self.skip_step = args.skip_step
        self.norm = args.norm

        # Won't change in general
        self.n_hidden = args.n_hidden
        self.n_negative = args.n_negative
        self.diffusion_steps = args.diffusion_steps
        # Sensitive Parameters
        self.uncondition_rate = args.uncondition_rate
        self.category_uncondition_rate = args.category_uncondition_rate
        self.tau = args.tau
        self.delta = args.delta

        self.lambda_rec_loss = args.lambda_rec_loss  # NOT IN THE PAPER
        self.lambda_mse_loss = args.lambda_mse_loss
        self.lambda_user_loss = args.lambda_user_loss

        # coefficient of prior
        scale = args.scale / args.diffusion_steps
        beta_start = scale * args.beta_start + args.beta_base
        beta_end = scale * args.beta_end + args.beta_base
        if beta_end > 1:
            beta_end = 1 / args.diffusion_steps + args.beta_base
        self.betas = torch.linspace(beta_start, beta_end, args.diffusion_steps, device=device)
        alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.alphas_bar_prev = torch.concat((torch.tensor([1.0], device=device), self.alphas_bar[:-1]))

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

        # coefficient of posterior
        self.p_mu_c1 = self.betas * torch.sqrt(self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.p_mu_c2 = (1.0 - self.alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_bar)
        self.p_sqrt_var = torch.sqrt(self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar))

        # Approximator
        ## Condition Encoding
        self.user_embeddings = torch.nn.Embedding(dataset.n_users, args.n_hidden)
        self.c_l_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden, args.n_hidden),
            torch.nn.Sigmoid()
        )

        ## Guidance Encoding
        ### Recent Interaction Sequence
        self.item_embeddings = torch.nn.Embedding(dataset.n_items + 1, args.n_hidden, padding_idx=dataset.n_items)
        self.transformer = TransformerEncoder(args.n_hidden, device, args.n_head, args.n_layer)
        self.none_embedding = torch.nn.Embedding(1, args.n_hidden)
        self.prototypes = torch.nn.Parameter(
            torch.randn(512, args.n_hidden, device=self.device)
        )
        ### Category Preference
        self.g_c_mlp = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout_g_c),
            torch.nn.Linear(len(self.dataset.cat2id), args.n_hidden),
            torch.nn.Sigmoid()
        )
        self.category_none_embedding = torch.nn.Embedding(1, args.n_hidden)

        ## Timestep Embedding
        self.timestep_embeddings = self.get_timestep_embeddings(torch.arange(args.diffusion_steps, device=device),
                                                                args.n_hidden)

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden, args.n_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(args.n_hidden, args.n_hidden)
        )
        self.time_embed1 = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden, args.n_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(args.n_hidden, args.n_hidden)
        )
        self.fusion_model = TimeAwareAttentionFusion(d_model=64)
        ## xt
        self.xt_mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(args.n_hidden, args.n_hidden, bias=True),
            torch.nn.Sigmoid(),
        )

        # Fusing Layer
        self.fusing_layer_intent = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden * 3, args.n_hidden),
            torch.nn.Sigmoid(),
        )
        self.fusing_layer_early = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden * 3, args.n_hidden),
            torch.nn.Sigmoid(),
        )
        self.fusing_layer_late = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden * 3, args.n_hidden),
            torch.nn.Sigmoid(),
        )

        # diffusion step sampler
        self.diffusion_step_sampler = create_named_schedule_sampler('lossaware', args.diffusion_steps)

        # diffusion mse loss weight
        self.mse_weight = (self.alphas_bar_prev / (1 - self.alphas_bar_prev) - self.alphas_bar / (
                1 - self.alphas_bar)) / 2

        # reverse steps
        self.reverse_steps = list(range(self.diffusion_steps))[::-self.skip_step]

        self.to(device)

    def distance(self, x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        dis = xx_cc - 2 * xc
        return dis

    # def distance(self, x, c):
    #     """
    #     x: (B, d)
    #     c: (K, d)
    #     返回 dis: (B, K)，表示每个样本与每个中心的欧式距离平方
    #     """
    #     # 扩展维度方便广播计算
    #     x_exp = x.unsqueeze(1)  # (B, 1, d)
    #     c_exp = c.unsqueeze(0)  # (1, K, d)

    #     # 计算逐元素平方差并在最后一维求和
    #     dis = ((x_exp - c_exp) ** 2).sum(dim=-1)  # (B, K)
    #     return dis

    def get_timestep_embeddings(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_item_embeddings(self, items):
        item_embeddings = self.item_embeddings(torch.LongTensor(items).to(self.device))
        if self.norm:
            item_embeddings = torch.nn.functional.normalize(item_embeddings, dim=-1)
        return item_embeddings,

    def get_original_user_embeddings(self, users):
        user_embeddings = self.user_embeddings(torch.LongTensor(users).to(self.device))
        if self.norm:
            user_embeddings = torch.nn.functional.normalize(user_embeddings, dim=-1)
        return user_embeddings

    def evaluate_x0(self, xt, t, intention, condition, category_condition, training):
        bs = xt.size(0)
        T = 32
        emb_t = self.time_embed(self.timestep_embeddings[t])
        emb_t1 = self.time_embed1(self.timestep_embeddings[t])
        if training:
            # Unconditional Version
            mask = (torch.rand(bs, device=self.device) <= self.uncondition_rate).float().view(bs, 1)
            condition = (1 - mask) * condition + mask * self.none_embedding(torch.tensor([0], device=self.device))

            category_mask = (torch.rand(bs, device=self.device) <= self.category_uncondition_rate).float().view(bs, 1)
            intention = (1 - category_mask) * intention + category_mask * self.category_none_embedding(
                torch.tensor([0], device=self.device))

        # x0_hat = self.fusing_layer(torch.concat((xt, condition, category_condition, user_condition, emb_t), dim=-1))

        attn, fused = self.fusion_model(intention, condition, t, T=self.diffusion_steps)
        cat_input = torch.cat((xt * self.delta, fused, emb_t), dim=-1)
        x0_hat = self.fusing_layer_intent(cat_input)
        # ---- 核心修改：根据时间步选择不同的融合层 ----
        # if isinstance(t, torch.Tensor):
        #     # 若是批量形式，例如 (B,)
        #     mask = (t >= self.n).float().unsqueeze(-1).to(cat_input.device)  # (B,1)

        #     # 两个层分别前向传播
        #     x0_hat_early = self.fusing_layer_early(cat_input)
        #     x0_hat_late = self.fusing_layer_late(cat_input)

        #     # 混合（按mask在每个样本上选择）
        #     x0_hat = x0_hat_early * (1 - mask) + x0_hat_late * mask
        # else:
        #     # 单步形式
        #     if t >= self.n:
        #         x0_hat = self.fusing_layer_late(cat_input)
        #     else:
        #         x0_hat = self.fusing_layer_early(cat_input)
        return  attn, x0_hat

    def get_user_embeddings(self, users, histories, lengths, x0=None, user_cat_hist=None, _=None):
        # User-specific Embedding -> Long-term Interest
        user_embeddings = self.get_original_user_embeddings(users)
        user_condition = self.c_l_mlp(user_embeddings)

        # Recent Interaction Sequence -> Short-term Interest
        history_embeddings = self.get_item_embeddings(histories)[0]
        lengths = torch.LongTensor(lengths).to(self.device)
        encoder_input = torch.cat([user_embeddings.unsqueeze(1), history_embeddings], dim=1)
        condition, intention = self.transformer.forward(encoder_input, lengths)

        # Category Distribution -> Category Preference
        if user_cat_hist is None:
            user_cat_hist = self.dataset.user_cat_hist

        user_cat_hist = user_cat_hist[torch.LongTensor(users).to(self.device)]
        category_condition = self.g_c_mlp(user_cat_hist)
        proto_exp = self.prototypes.unsqueeze(0).expand(intention.shape[0], -1, -1)

        # intention = intention.unsqueeze(1)

        # # 计算相似度（点积或余弦等）
        # sim = torch.sum(proto_exp * intention, dim=-1)  # (B, K)
        # idx = torch.argmax(sim, dim=-1)  # (B,)
        # alpha = F.one_hot(idx, num_classes=self.prototypes.shape[0]).float()  # (B, K)
        # z_c = torch.matmul(alpha, self.prototypes)
        z_c = intention
        bs = len(histories)
        if x0 is not None:
            t, _ = self.diffusion_step_sampler.sample(bs, self.device)
            noise = torch.randn_like(x0)
            xt = self.sqrt_alphas_bar[t].unsqueeze(-1) * x0 + self.sqrt_one_minus_alphas_bar[t].unsqueeze(-1) * noise
            attn, x0_hat = self.evaluate_x0(xt, t, z_c, condition, category_condition, training=True)
        else:
            noise_xt = torch.rand((bs, self.n_hidden), device=self.device)
            attn_0_list = []
            attn_1_list = []
            for step in self.reverse_steps:
                t = torch.tensor([step] * bs, device=self.device)
                attn, x0_hat = self.evaluate_x0(noise_xt, t, z_c, condition, category_condition, training=False)
                attn_0_list.append(attn[:,0].mean().item())
                attn_1_list.append(attn[:,1].mean().item())
                model_mean = self.p_mu_c1[t].unsqueeze(-1) * x0_hat + self.p_mu_c2[t].unsqueeze(-1) * noise_xt
                model_sqrt_var = self.p_sqrt_var[t].unsqueeze(-1)
                noise = torch.randn_like(noise_xt)
                noise_xt = model_mean + model_sqrt_var * noise
            print(attn_0_list)
            print(attn_1_list)
            # plt.plot(range(32), attn_0_list, label='attn[:,0]')
            # plt.plot(range(32), attn_1_list, label='attn[:,1]')
            # plt.xlabel('t')
            # plt.ylabel('Attention Weight')
            # plt.title('Attention Weight vs Time Step')
            # plt.legend()
            # plt.show()
            x0_hat = noise_xt
        return x0_hat, t, user_embeddings, z_c

    def forward_bpr(self, users, histories, lengths, pos_items):
        bs = len(pos_items)
        neg_items = torch.randint(0, self.dataset.n_items, (bs, self.n_negative))  # bs x n_negative
        pos_x0 = self.get_item_embeddings(pos_items)[0]
        neg_x0_list = self.get_item_embeddings(neg_items)[0]  # bs x n_negative x d

        # x0_hat, t, user_embeddings = self.get_user_embeddings(users, histories, lengths, pos_x0, x0_id=pos_items)
        x0_hat, t, user_embeddings, z_c = self.get_user_embeddings(users, histories, lengths, pos_x0)

        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        L_cluster = self.distance(prototypes, prototypes)
        mask = 1 - torch.eye(prototypes.shape[0], device=prototypes.device)
        L_cluster = (L_cluster * mask).sum() / (prototypes.shape[0] * (prototypes.shape[0] - 1))
        L_cluster = -L_cluster.mean() * 0

        # dis = self.distance(self.prototypes,self.prototypes)
        # L_cluster = L_cluster.mean()
        # L_cluster = - dis / dis.detach().mean()

        # L_recon = self.distance(pos_x0, z_c).mean()
        # User Loss
        pos_user_loss = -torch.mean(torch.log(torch.sigmoid(torch.sum(z_c * pos_x0, dim=-1)) + 1e-24))
        neg_user_score = torch.sum(z_c.view(bs, 1, -1) * neg_x0_list, dim=-1)  # bs x n_negative
        neg_user_loss = - torch.mean(torch.sum(torch.log(torch.sigmoid(-neg_user_score) + 1e-24), dim=-1))
        L_recon = pos_user_loss+ neg_user_loss

        # Recommendation Loss
        tau_coeff = torch.where(t <= self.tau, 1.0, 0.0)
        # pos_loss = -torch.mean(tau_coeff * torch.log(torch.sigmoid(torch.sum(x0_hat * user_embeddings, dim=-1)) + 1e-24))  # TODO: !important
        pos_loss = -torch.mean(tau_coeff * torch.log(torch.sigmoid(torch.sum(x0_hat * pos_x0, dim=-1)) + 1e-24))
        neg_score = torch.sum(x0_hat.view(bs, 1, -1) * neg_x0_list, dim=-1)  # bs x n_negative
        neg_loss = -torch.mean(torch.sum(tau_coeff.view(bs, 1) * torch.log(torch.sigmoid(-neg_score) + 1e-24), dim=-1))
        loss = pos_loss+ neg_loss

        # Diffusion Loss
        x0_ = pos_x0
        # x0_ = x0.detach()
        mse = torch.mean((x0_ - x0_hat) ** 2, dim=-1)
        weight = torch.where((t == 0), 1.0, self.mse_weight[t])
        mse_loss = torch.mean(mse * weight)

        return L_cluster * 0.5, L_recon*0.05, loss*0.8, mse_loss*1.2
