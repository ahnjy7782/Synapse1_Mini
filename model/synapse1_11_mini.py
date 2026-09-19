from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_layers import GradientCheckpointingLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

_FLASH_HEAD_DIMS = frozenset({16, 32, 64, 96, 128})
_warned_head_dims = set()


class SynapseBrainConfig(PretrainedConfig):
    model_type = "Synapse1d11Mini"
    # 1.11 ver
    # SWA/GQA 사용, 직병렬 구조 사용, Multi Layer LM Head(아이디어) 구현 완료

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        hidden_size=480,                # 1024
        intermediate_size=1920,          # 1200
        sliding_window=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_hidden_layers=18,
        max_position_embeddings=2048,
        block_scale_init=0.01,

        residual_scale_init=0.35,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="situ-glu",
        weight_decay=0.03,
        resid_pdrop=0.05,
        embd_pdrop=0.05,
        attn_pdrop=0.05,
        layer_norm_eps=1e-5,
        initializer_range=0.025,
        use_qk_norm=True,
        attn_scale_mode="per_head",
        head_dim=None,
        attn_implementation="sdpa",
        use_parallel_residual=False,
        # tie_word_embeddings=True,
        token_drop_rate=0.0,
        token_drop_layers=None,
        xsa_every_n_layers=2,  # 0=no XSA, 1=all layers use XSA-GQA, 2=alternate GQA/XSA-GQA
        unk_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        **kwargs
    ):
        super().__init__(
            unk_token_id=unk_token_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            # tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.sliding_window = sliding_window
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for GQA"
            )
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.block_scale_init = block_scale_init
        self.residual_scale_init = residual_scale_init
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_act = hidden_act
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range if initializer_range is not None else 0.02 / math.sqrt(2 * config.num_hidden_layers)
        self.weight_decay = weight_decay
        self.use_qk_norm = use_qk_norm
        self.attn_scale_mode = attn_scale_mode
        self.head_dim = head_dim
        self.attn_implementation = attn_implementation
        self.use_parallel_residual = use_parallel_residual
        self.token_drop_rate = token_drop_rate
        self.token_drop_layers = token_drop_layers if token_drop_layers is not None else [1, -1]
        self.xsa_every_n_layers = xsa_every_n_layers


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        x_normed = x_normed.to(orig_dtype)
        if self.bias is not None:
            return self.weight * x_normed + self.bias
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, theta=10000.0, rope_scaling=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.rope_scaling = rope_scaling
        self.original_max_seq_len = max_position_embeddings

        if rope_scaling is not None:
            rope_type = rope_scaling.get("type", None)
            if rope_type == "linear":
                self.max_position_embeddings = int(max_position_embeddings * rope_scaling.get("factor", 1.0))
                theta = theta * rope_scaling.get("factor", 1.0)
                
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        inv_freq = self.inv_freq
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("type", None)
            if rope_type == "dynamic":
                seq_len = position_ids.max().item() + 1
                if seq_len > self.original_max_seq_len:
                    factor = self.rope_scaling.get("factor", 1.0)
                    scaling_factor = (seq_len / self.original_max_seq_len) / factor
                    scaling_factor = max(1.0, scaling_factor)
                    new_theta = self.theta * scaling_factor
                    inv_freq = 1.0 / (new_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=position_ids.device) / self.dim))

        freqs = position_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    b, n_kv, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class FactorizedEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_proj = nn.Linear(config.embed_dim, config.hidden_size, bias=False) if config.embed_dim != config.hidden_size else nn.Identity()
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        x = self.embed_proj(x)
        return self.drop(x)


class BrainAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        self.head_dim = config.head_dim if config.head_dim is not None else config.hidden_size // config.num_attention_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_kv_heads}) for GQA"
            )
        self.n_rep = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_idx

        if torch.cuda.is_available() and self.head_dim not in _FLASH_HEAD_DIMS:
            if self.head_dim not in _warned_head_dims:
                _warned_head_dims.add(self.head_dim)
                warnings.warn(
                    f"SDPA: head_dim={self.head_dim} not in flash-attention supported sizes "
                    f"{sorted(_FLASH_HEAD_DIMS)}; attention falls back to mem-efficient/math kernels",
                    stacklevel=2,
                )

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

            
        if config.attn_scale_mode == "none":
            self.attn_log_scale = None

        elif config.attn_scale_mode == "global":
            self.attn_log_scale = nn.Parameter(torch.zeros(()))

        elif config.attn_scale_mode == "per_head":
            self.attn_log_scale = nn.Parameter(torch.zeros(self.num_heads))
        else:
            raise ValueError(f"attn_scale_mode must be 'none', 'global', or 'per_head', got {config.attn_scale_mode}")

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, attention_mask=None, past_key_values=None, layer_idx=0, position_ids=None, cache_position=None, use_xsa=False, is_causal=False):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.q_norm is not None:
            q = self.q_norm(q.float()).to(q.dtype)
            k = self.k_norm(k.float()).to(k.dtype)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        # XSA: 확장되지 않은 KV 캐시에서 V를 가져옴
        if use_xsa:
            v_curr = v[:, :, -T:, :]
            if self.n_rep > 1:
                v_curr = v_curr.repeat_interleave(self.n_rep, dim=1)
                
        if self.attn_log_scale is not None:
            if self.attn_log_scale.ndim == 0:
                scale = torch.exp(self.attn_log_scale).clamp(0.25, 4.0)
            else:
                scale = torch.exp(
                    self.attn_log_scale
                ).clamp(0.25, 4.0).view(1, self.num_heads, 1, 1)

            q = q * scale

            
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )

        # XSA: 어텐션 출력에서 V 방향 제거
        if use_xsa:
            vn = F.normalize(v_curr, dim=-1, eps=1e-6)
            y = y - 0.15 * (y * vn).sum(dim=-1, keepdim=True) * vn

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.out_proj(y))


class SWAAttention(nn.Module):
    """Sliding-window attention: each query attends to at most the last `window_size`
    keys (including itself), so sequence cost is O(T * window_size) instead of O(T^2)."""

    def __init__(self, config, layer_idx=0, window_size=512):
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = getattr(config, "sliding_window", window_size)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        self.head_dim = config.head_dim if config.head_dim is not None else config.hidden_size // config.num_attention_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_kv_heads}) for GQA"
            )
        self.n_rep = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_idx

        if torch.cuda.is_available() and self.head_dim not in _FLASH_HEAD_DIMS:
            if self.head_dim not in _warned_head_dims:
                _warned_head_dims.add(self.head_dim)
                warnings.warn(
                    f"SDPA: head_dim={self.head_dim} not in flash-attention supported sizes "
                    f"{sorted(_FLASH_HEAD_DIMS)}; attention falls back to mem-efficient/math kernels",
                    stacklevel=2,
                )

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if config.attn_scale_mode == "none":
            self.attn_log_scale = None
        elif config.attn_scale_mode == "global":
            self.attn_log_scale = nn.Parameter(torch.zeros(()))
        elif config.attn_scale_mode == "per_head":
            self.attn_log_scale = nn.Parameter(torch.zeros(self.num_heads))
        else:
            raise ValueError(f"attn_scale_mode must be 'none', 'global', or 'per_head', got {config.attn_scale_mode}")

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _build_swa_mask(self, T, total_len, past_len, dtype, device):
        w = self.window_size
        causal = torch.triu(
            torch.full((T, total_len), float("-inf"), device=device, dtype=dtype),
            diagonal=past_len + 1,
        )
        window = torch.tril(
            torch.full((T, total_len), float("-inf"), device=device, dtype=dtype),
            diagonal=past_len - w,
        )
        return causal + window

    def forward(self, x, attention_mask=None, past_key_values=None, layer_idx=0, position_ids=None, cache_position=None, use_xsa=False):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.q_norm is not None:
            q = self.q_norm(q.float()).to(q.dtype)
            k = self.k_norm(k.float()).to(k.dtype)

        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        # XSA: 확장되지 않은 KV 캐시에서 V를 가져옴
        if use_xsa:
            v_curr = v[:, :, -T:, :]
            if self.n_rep > 1:
                v_curr = v_curr.repeat_interleave(self.n_rep, dim=1)

        if self.attn_log_scale is not None:
            if self.attn_log_scale.ndim == 0:
                scale = torch.exp(self.attn_log_scale).clamp(0.25, 4.0)
            else:
                scale = torch.exp(
                    self.attn_log_scale
                ).clamp(0.25, 4.0).view(1, self.num_heads, 1, 1)

            q = q * scale

        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)

        total_len = past_len + T
        mask = self._build_swa_mask(T, total_len, past_len, x.dtype, x.device)
        mask = mask[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, total_len]
                pad = attention_mask[:, None, None, :total_len].to(dtype=x.dtype)
                pad = (1.0 - pad) * torch.finfo(x.dtype).min

            elif attention_mask.dim() == 4:
                pad = attention_mask[:, :, -T:, :total_len].to(dtype=x.dtype)
                pad = (1.0 - pad) * torch.finfo(x.dtype).min

            else:
                raise ValueError(
                    f"Unsupported attention_mask shape: {attention_mask.shape}"
                )

            mask = mask + pad

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )

        # XSA: 어텐션 출력에서 V를 제거
        if use_xsa:
            vn = F.normalize(v_curr, dim=-1, eps=1e-6)
            y = y - 0.15 * (y * vn).sum(dim=-1, keepdim=True) * vn

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.out_proj(y))


# 기존 SwiGLU -> SiTU-GLU로 대체
# 목적: 안정화
class SiTUGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.beta_gate = 4.0
        self.beta_up = 25.0

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        gate = self.beta_gate * torch.tanh(gate / self.beta_gate) * torch.sigmoid(gate)
        up = self.beta_up * torch.tanh(up / self.beta_up)
        return self.down_proj(gate * up)


class BrainBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_full_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        if layer_idx % 3 == 2:  #   GQA -> GQA -> SWA   2:1
            # SWA을 만들어 놓고도 안 쓴 멍충이!
            self.attn = SWAAttention(config, layer_idx, config.sliding_window)
        else:
            self.attn = BrainAttention(config, layer_idx)
        #self.mixgate = MixGate(config)
        self.ffn = SiTUGLU(config)

        scale = config.residual_scale_init
        self.attn_res_scale = nn.Parameter(torch.tensor(scale))
        self.ffn_res_scale = nn.Parameter(torch.tensor(scale))

        self.block_scale = nn.Parameter(torch.tensor(config.block_scale_init))
        self.block_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask=NotImplemented, past_key_values=None, layer_idx=0, position_ids=None, cache_position=None, use_xsa=False):
        if layer_idx % 3 == 2:
            ah = self.attn_norm(x)
            fh = self.ffn_norm(x)

            x = (
                x 
                + self.ffn_res_scale * self.ffn(fh)
                + self.attn_res_scale * (
                    self.attn(ah, attention_mask, past_key_values, layer_idx, position_ids, cache_position, use_xsa=use_xsa)
                )
            )
        else:
            ah = self.attn_norm(x)

            x = x + self.attn_res_scale * (
                self.attn(ah, attention_mask, past_key_values, layer_idx, position_ids, cache_position, use_xsa=use_xsa)
            )

            fh = self.ffn_norm(x)
            x = x + self.ffn_res_scale * self.ffn(fh)

        block_state = self.block_scale * self.block_norm(x)
        # 기획한 Multi Layer LM Head는 실제로 효과가 있음.
        # 학습 시 block_scale이 서로 다르며, 
        # 초반 레이어는 음수~0.01(미미한 흐름),     # 여기서 특이한 점은 음수가 매우 잘 나타난다는 점. 토큰이 아직 의미를 형성하는 단계라 그런 듯
                                                # 1~4 레이어는 대체로 -1.1 ~ -0.6 임.
                                                # 아무튼 해당 레이어의 출력 만큼 제거하는 것이 LM Head에게 좋은 결과를 주는 듯 한.
                                                # 적은 파라미터로 성능을 조금 향상 시킨! 행복!
        # 중반 레이어는 scale > 0.7
        # 후반 레이어는 2 > scale > 0.6
        # 이런 형태를 보임.

        # 만약 이 코드 보시면 학습할 땐 실행 안 하는 것을 추천.
        # Generate에서 실행해 보시면 좋고, 변화하는 것을 보고 싶으시다면 1 epochs에 데이터는 작게 하시면 됩니다.
        # print(f"{layer_idx} : ", self.block_scale.detach())

        return x, block_state


class SynapseBrain(PreTrainedModel, GenerationMixin):
    config_class = SynapseBrainConfig
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["BrainBlock"]
    _keys_to_ignore_on_load_missing = []
    # _tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.embedding = FactorizedEmbedding(config)
        self.blocks = nn.ModuleList([BrainBlock(config, i) for i in range(config.num_hidden_layers)])

        self.final_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.hidden_to_embed = nn.Linear(config.hidden_size, config.embed_dim, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.embed_tokens

    def set_input_embeddings(self, value):
        self.embedding.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head     # self.embedding.embed_tokens

    def set_output_embeddings(self, new_embeddings):
        # self.embedding.embed_tokens = new_embeddings
        self.lm_head = new_embeddings

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        # elif isinstance(module, nn.Conv1d):       # 현재 모델에서 Conv가 제거 됨
        #     module.weight.data.normal_(mean=0.0, std=std * 0.45)

    @staticmethod
    def _prepare_4d_causal_mask(attention_mask, T, total_len, past_len, dtype, device):
        # Always build the causal mask — never return None.
        # This ensures SDPA never falls back to is_causal=True, which is
        # incorrect when KV cache shifts the Q/K alignment.
        causal = torch.triu(
            torch.full((T, total_len), float("-inf"), device=device, dtype=dtype),
            diagonal=past_len + 1,
        )
        causal = causal[None, None, :, :]
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :total_len].to(dtype=dtype)
            pad = (1.0 - pad) * torch.finfo(dtype).min
            causal = causal + pad
        return causal
    

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=None, cache_position=None, inputs_embeds=None, **kwargs):
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache()

        if inputs_embeds is not None:
            x = self.embedding.drop(inputs_embeds)
        else:
            x = self.embedding(input_ids)

        T = x.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        total_len = past_len + T

        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0) if cache_position.dim() == 1 else cache_position
        else:
            position_ids = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(0)

        mask = self._prepare_4d_causal_mask(attention_mask, T, total_len, past_len, x.dtype, x.device)

        n_layers = self.config.num_hidden_layers
        drop_rate = self.config.token_drop_rate
        drop_start, drop_end = self.config.token_drop_layers
        if drop_end < 0:
            drop_end = n_layers + drop_end

        xsa_every_n = self.config.xsa_every_n_layers
        block_state = []
        for i in range(n_layers):
            block = self.blocks[i]
            use_xsa = (xsa_every_n > 0) and (i % xsa_every_n == 0)
            x, state = block(x, mask, past_key_values, i, position_ids, cache_position, use_xsa=use_xsa)

            if state is not None:
                block_state.append(state)

            if self.training and drop_rate > 0.0 and drop_start <= i < drop_end:
                if i < n_layers - 1:
                    B_i, T_i = x.shape[:2]
                    keep_prob = 1.0 - drop_rate
                    keep_mask = (
                        torch.rand(B_i, T_i, 1, device=x.device) < keep_prob
                    ).to(x.dtype)
                    x = x * keep_mask / keep_prob

        x = self.final_norm(sum(block_state))
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

            # aux_loss = self.get_moe_aux_loss()
            loss = lm_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    config = SynapseBrainConfig()
    model = SynapseBrain(config)
    model.eval()

    
    from torchinfo import summary
    summary(
        model,
        input_size=(1, 64),
        dtypes=[torch.long]
    )

    from embedding_layer import analyze_embedding_ratio
    analyze_embedding_ratio(model)

    model_compiled = model
    # try:
    #     model_compiled = torch.compile(model, fullgraph=True, mode="max-autotune")
    #     out_c = model_compiled(x)
    #     print(f"Compiled forward OK, logits shape: {out_c.logits.shape}")
    # except Exception as e:
    #     print(f"Compile skipped ({e})")

    tokenizer = AutoTokenizer.from_pretrained("_tokenizer\\model\\synapse_tokenizer_bpe_v3")
    model_compiled.save_pretrained("synapse1_mini")
    tokenizer.save_pretrained("synapse1_mini")