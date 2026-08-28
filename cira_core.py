import ast
import importlib.util
import json
import os
import re
import string
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, CLIPModel, LlamaConfig


ROOT = os.path.dirname(os.path.abspath(__file__))
LLAVA_ROOT = os.path.join(ROOT, "LLaVA")
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TEXTVQA_INSTRUCTION = "Answer the question using a single word or a short phrase."
_ANSWER_PROCESSOR = None
_LLAVA_LOADED = False

CLIPAttention = None
CLIPEncoderLayer = None
CLIPVisionTower = None
CLIPVisionTower_VisionZip = None
LlavaMetaForCausalLM = None
DEFAULT_IMAGE_TOKEN = None
DEFAULT_IM_END_TOKEN = None
DEFAULT_IM_START_TOKEN = None
IMAGE_TOKEN_INDEX = None
conv_templates = None
disable_torch_init = None
encode_images_visionzip = None
encode_images_visionzip_multi = None
get_model_name_from_path = None
load_pretrained_model = None
prepare_inputs_labels_for_multimodal_visionzip = None
process_images = None
restore_image_features_sorted = None
tokenizer_image_token = None
visionzip = None
CLIPAttention_forward = None
CLIP_EncoderLayer_forward = None

_ORIG_CLIP_LAYER = None
_ORIG_CLIP_ATTN = None
_ORIG_CLIP_TOWER = None
_ORIG_PREPARE = None


def set_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_budget_pairs(value):
    text = (value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            parsed = [parsed]
        if isinstance(parsed, list):
            pairs = []
            for pair in parsed:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError
                dominant, contextual = int(pair[0]), int(pair[1])
                if dominant <= 0 or contextual < 0:
                    raise ValueError
                pairs.append((dominant, contextual))
            return pairs
    except (SyntaxError, TypeError, ValueError):
        pass
    values = [int(item) for item in re.findall(r"-?\d+", text)]
    if len(values) % 2:
        raise ValueError("budgets must contain pairs of integers")
    pairs = list(zip(values[::2], values[1::2]))
    if any(dominant <= 0 or contextual < 0 for dominant, contextual in pairs):
        raise ValueError("budget values must be positive")
    return pairs


def patch_llama_config(config):
    defaults = LlamaConfig().to_dict()
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    return config


def _image_stats(processor, reference):
    mean = torch.tensor(
        processor.image_mean, device=reference.device, dtype=reference.dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        processor.image_std, device=reference.device, dtype=reference.dtype
    ).view(1, 3, 1, 1)
    return mean, std


def normalize_image(image, processor):
    mean, std = _image_stats(processor, image)
    return (image - mean) / std


def denormalize_image(image, processor):
    mean, std = _image_stats(processor, image)
    return (image * std + mean).clamp(0.0, 1.0)


def clip_normalize(image):
    mean = torch.tensor(CLIP_MEAN, device=image.device, dtype=image.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(CLIP_STD, device=image.device, dtype=image.dtype).view(
        1, 3, 1, 1
    )
    return (image - mean) / std


def project_linf_image(clean, candidate, epsilon):
    delta = (candidate - clean).clamp(-float(epsilon), float(epsilon))
    return (clean + delta).clamp(0.0, 1.0)


def parse_attention_layers(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    layers = tuple(int(item) for item in re.findall(r"-?\d+", str(value)))
    if not layers:
        raise ValueError("invalid attention layer specification")
    return layers


def cls_patch_attention_score(attentions, layers=(-2,), head_reduce="sum"):
    if not attentions:
        raise ValueError("attention output is empty")
    count = len(attentions)
    scores = []
    for layer in layers:
        index = int(layer) if int(layer) >= 0 else count + int(layer)
        if index < 0 or index >= count:
            raise ValueError("attention layer is out of range")
        values = attentions[index][:, :, 0, 1:]
        if head_reduce == "sum":
            values = values.sum(dim=1)
        elif head_reduce == "mean":
            values = values.mean(dim=1)
        else:
            raise ValueError("unsupported head reduction")
        scores.append(values.float())
    return torch.stack(scores).mean(dim=0)


def _load_llava():
    global CLIPAttention, CLIPEncoderLayer
    global CLIPVisionTower, CLIPVisionTower_VisionZip
    global LlavaMetaForCausalLM
    global DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    global IMAGE_TOKEN_INDEX, conv_templates, disable_torch_init
    global encode_images_visionzip, encode_images_visionzip_multi
    global get_model_name_from_path, load_pretrained_model
    global prepare_inputs_labels_for_multimodal_visionzip, process_images
    global restore_image_features_sorted, tokenizer_image_token, visionzip
    global CLIPAttention_forward, CLIP_EncoderLayer_forward
    global _ORIG_CLIP_LAYER, _ORIG_CLIP_ATTN
    global _ORIG_CLIP_TOWER, _ORIG_PREPARE, _LLAVA_LOADED

    if _LLAVA_LOADED:
        return

    from transformers.models.clip.modeling_clip import (
        CLIPAttention as _CLIPAttention,
        CLIPEncoderLayer as _CLIPEncoderLayer,
    )
    from VisionZip.visionzip import visionzip as _visionzip
    from VisionZip.visionzip.clip_encoder import (
        CLIPVisionTower_VisionZip as _CLIPVisionTower_VisionZip,
    )
    from VisionZip.visionzip.llava_arch import (
        encode_images_visionzip as _encode_images_visionzip,
        encode_images_visionzip_multi as _encode_images_visionzip_multi,
        prepare_inputs_labels_for_multimodal_visionzip as _prepare_inputs,
        restore_image_features_sorted as _restore_image_features_sorted,
    )
    from VisionZip.visionzip.utils import (
        CLIPAttention_forward as _CLIPAttention_forward,
        CLIP_EncoderLayer_forward as _CLIP_EncoderLayer_forward,
    )
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN as _DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN as _DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN as _DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX as _IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates as _conv_templates
    from llava.mm_utils import (
        get_model_name_from_path as _get_model_name_from_path,
        process_images as _process_images,
        tokenizer_image_token as _tokenizer_image_token,
    )
    from llava.model.builder import (
        load_pretrained_model as _load_pretrained_model,
    )
    from llava.model.llava_arch import (
        LlavaMetaForCausalLM as _LlavaMetaForCausalLM,
    )
    from llava.model.multimodal_encoder.clip_encoder import (
        CLIPVisionTower as _CLIPVisionTower,
    )
    from llava.utils import disable_torch_init as _disable_torch_init

    CLIPAttention = _CLIPAttention
    CLIPEncoderLayer = _CLIPEncoderLayer
    CLIPVisionTower = _CLIPVisionTower
    CLIPVisionTower_VisionZip = _CLIPVisionTower_VisionZip
    LlavaMetaForCausalLM = _LlavaMetaForCausalLM
    DEFAULT_IMAGE_TOKEN = _DEFAULT_IMAGE_TOKEN
    DEFAULT_IM_END_TOKEN = _DEFAULT_IM_END_TOKEN
    DEFAULT_IM_START_TOKEN = _DEFAULT_IM_START_TOKEN
    IMAGE_TOKEN_INDEX = _IMAGE_TOKEN_INDEX
    conv_templates = _conv_templates
    disable_torch_init = _disable_torch_init
    encode_images_visionzip = _encode_images_visionzip
    encode_images_visionzip_multi = _encode_images_visionzip_multi
    get_model_name_from_path = _get_model_name_from_path
    load_pretrained_model = _load_pretrained_model
    prepare_inputs_labels_for_multimodal_visionzip = _prepare_inputs
    process_images = _process_images
    restore_image_features_sorted = _restore_image_features_sorted
    tokenizer_image_token = _tokenizer_image_token
    visionzip = _visionzip
    CLIPAttention_forward = _CLIPAttention_forward
    CLIP_EncoderLayer_forward = _CLIP_EncoderLayer_forward

    _ORIG_CLIP_LAYER = CLIPEncoderLayer.forward
    _ORIG_CLIP_ATTN = CLIPAttention.forward
    _ORIG_CLIP_TOWER = CLIPVisionTower.forward
    _ORIG_PREPARE = LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal
    _LLAVA_LOADED = True


class LlavaVisionZipRunner:
    def __init__(
        self,
        model_path,
        vision_tower=None,
        model_base=None,
        dominant=27,
        contextual=5,
        device=None,
    ):
        _load_llava()
        disable_torch_init()
        name = get_model_name_from_path(model_path)
        load_kwargs = {}
        if vision_tower:
            config = patch_llama_config(AutoConfig.from_pretrained(model_path))
            config.mm_vision_tower = vision_tower
            load_kwargs["config"] = config
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=name,
            **load_kwargs,
        )
        self.device = self.model.device if device is None else torch.device(device)
        self.model.eval()
        self.current_budget = (int(dominant), int(contextual))
        self.visionzip_initialized = False
        self.base_conv = conv_templates["llava_v1"].copy()
        self.use_visionzip()

    def use_llava(self):
        CLIPEncoderLayer.forward = _ORIG_CLIP_LAYER
        CLIPAttention.forward = _ORIG_CLIP_ATTN
        CLIPVisionTower.forward = _ORIG_CLIP_TOWER
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = _ORIG_PREPARE

    def use_visionzip(self):
        if not self.visionzip_initialized:
            self.model = visionzip(
                self.model,
                dominant=self.current_budget[0],
                contextual=self.current_budget[1],
            )
            self.visionzip_initialized = True
            return
        CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
        CLIPAttention.forward = CLIPAttention_forward
        CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
            prepare_inputs_labels_for_multimodal_visionzip
        )
        LlavaMetaForCausalLM.restore_image_features_sorted = (
            restore_image_features_sorted
        )
        LlavaMetaForCausalLM.encode_images_visionzip_multi = (
            encode_images_visionzip_multi
        )
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip

    def set_visionzip_budget(self, dominant, contextual):
        self.use_visionzip()
        tower = self.model.model.vision_tower.vision_tower
        tower._info["dominant"] = int(dominant) - 1
        tower._info["contextual"] = int(contextual)
        self.current_budget = (int(dominant), int(contextual))

    def build_prompt(self, question):
        conversation = self.base_conv.copy()
        if getattr(self.model.config, "mm_use_im_start_end", False):
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            prompt = DEFAULT_IMAGE_TOKEN
        conversation.append_message(conversation.roles[0], prompt + "\n" + question)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()

    def load_image_tensor(self, path):
        image = Image.open(path).convert("RGB")
        tensor = process_images([image], self.image_processor, self.model.config)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device=self.device, dtype=torch.float16)

    def tensor_to_image01(self, tensor):
        return denormalize_image(tensor.float(), self.image_processor)

    def image01_to_tensor(self, image):
        return normalize_image(image.float(), self.image_processor).to(
            device=self.device, dtype=torch.float16
        )

    def predict_tensor(self, image, question, budget=None, max_tokens=64):
        if budget is None:
            self.use_llava()
        else:
            self.set_visionzip_budget(*budget)
        input_ids = (
            tokenizer_image_token(
                self.build_prompt(question),
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.device)
        )
        image = image.to(device=self.device, dtype=torch.float16)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                images=image,
                do_sample=False,
                max_new_tokens=max_tokens,
                use_cache=True,
            )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


def final_vision_tokens(model, output):
    features = output.last_hidden_state
    norm = getattr(model.vision_model, "post_layernorm", None)
    return norm(features) if norm is not None else features


def clip_vision_forward(model, pixel):
    _load_llava()
    layer_forward = CLIPEncoderLayer.forward
    attention_forward = CLIPAttention.forward
    CLIPEncoderLayer.forward = _ORIG_CLIP_LAYER
    CLIPAttention.forward = _ORIG_CLIP_ATTN
    try:
        return model.vision_model(
            pixel, output_hidden_states=True, output_attentions=True
        )
    finally:
        CLIPEncoderLayer.forward = layer_forward
        CLIPAttention.forward = attention_forward


def get_clip_feats_scores(
    model, image, sel_layer=-2, score_method="attn", attn_score_layers=(-2,)
):
    output = clip_vision_forward(model, clip_normalize(image.float()))
    features = final_vision_tokens(model, output)
    if score_method == "norm":
        scores = features[:, 1:].norm(dim=-1)
    elif score_method == "attn":
        scores = cls_patch_attention_score(output.attentions, attn_score_layers)
    else:
        raise ValueError("score_method must be 'norm' or 'attn'")
    return features, scores


def ranks_from_scores(scores, descending=True):
    order = torch.argsort(scores, dim=1, descending=descending)
    ranks = torch.empty_like(order)
    values = torch.arange(1, scores.shape[1] + 1, device=scores.device).view(1, -1)
    ranks.scatter_(1, order, values.expand_as(order))
    return ranks


def target_scores_from_clean_ranks(ranks, gamma=1.0, like=None):
    count = ranks.shape[-1]
    scores = ((ranks.float() - 1) / max(count - 1, 1)).clamp(0, 1)
    scores = scores.sqrt().pow(float(gamma))
    return (
        scores.to(device=like.device, dtype=like.dtype) if like is not None else scores
    )


def soft_ranks_from_scores(scores, eps=1e-6):
    values = torch.nan_to_num(scores.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    mean = values.mean(dim=-1, keepdim=True).detach()
    std = values.std(dim=-1, keepdim=True, unbiased=False).detach().clamp_min(eps)
    z = (values - mean) / std
    pairwise = torch.sigmoid(z.unsqueeze(1) - z.unsqueeze(2))
    eye = torch.eye(scores.shape[-1], device=scores.device, dtype=torch.bool).unsqueeze(
        0
    )
    return 1.0 + pairwise.masked_fill(eye, 0).sum(dim=-1)


def budget_survival_from_scores(scores, budget, eps=1e-6):
    count = scores.shape[-1]
    low, high = sorted((int(budget[0]), int(budget[1])))
    low = max(1, min(low, count))
    high = max(1, min(high, count))
    ranks = soft_ranks_from_scores(scores, eps)
    raw = (float(high + 1) - ranks) / max(float(high - low + 1), eps)
    raw = torch.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    clipped = raw.clamp(0.0, 1.0)
    return raw + (clipped - raw).detach()


def normalize_mass(mass, eps=1e-6):
    mass = torch.nan_to_num(mass.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(eps)


def route_prior(clean, target, eps=1e-6):
    clean = clean.float().clamp(0, 1)
    target = target.to(clean).clamp(0, 1)
    return normalize_mass(target * F.relu(target - clean), eps).detach()


def route_loss(adv, target, prior, log_eps=1e-3):
    adv = adv.float().clamp(0, 1)
    target = target.to(adv).clamp(0, 1)
    ratio = torch.log(adv + log_eps) - torch.log(target + log_eps)
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.log(adv.new_tensor((1 + log_eps) / log_eps)).clamp_min(1e-12)
    return -(prior.to(adv) * (ratio / scale).pow(2)).sum(dim=-1).mean()


def tda_loss(
    clean_features,
    adv_features,
    clean_survival,
    target_survival,
    adv_survival,
    prior,
    eps=1e-6,
    return_stats=False,
):
    clean = F.normalize(clean_features.float(), dim=-1, eps=1e-12).detach()
    adv = F.normalize(adv_features.float(), dim=-1, eps=1e-12)
    clean_survival = clean_survival.to(adv).clamp(0, 1)
    target_survival = target_survival.to(adv).clamp(0, 1)
    adv_survival = adv_survival.to(adv).clamp(0, 1)
    prior = prior.to(adv)
    gate = (
        ((adv_survival - clean_survival) / (target_survival - clean_survival + eps))
        .clamp(0, 1)
        .detach()
    )
    expose = prior * gate
    delivery = expose.sum(dim=-1)
    q_expose = normalize_mass(expose, eps).detach()
    q_full = normalize_mass(clean_survival, eps).detach()
    damage = 1 - torch.einsum("bnc,bnc->bn", adv, clean).clamp(-1, 1)
    expose_damage = (q_expose * damage).sum(dim=-1)
    full_damage = (q_full * damage).sum(dim=-1)
    value = (delivery.detach() * (expose_damage - full_damage)).mean()
    if not return_stats:
        return value
    return value, {
        "delivery": delivery.detach().mean(),
        "expose_damage": expose_damage.detach().mean(),
        "full_damage": full_damage.detach().mean(),
        "tda_gap": (expose_damage - full_damage).detach().mean(),
    }


@dataclass
class CIRAConfig:
    epsilon: float = 4 / 255
    alpha: float = 1 / 255
    steps: int = 100
    k_min: int = 32
    k_max: int = 192
    lambda_route: float = 1.0
    lambda_tda: float = 0.0
    rank_reverse_gamma: float = 1.0
    score_align_eps: float = 1e-6
    log_eps: float = 1e-3
    sel_layer: int = -2
    score_method: str = "attn"
    attn_score_layers: tuple = (-2,)
    random_init: bool = False


class CIRAAttacker:
    def __init__(self, model_id, config, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            self.model = CLIPModel.from_pretrained(
                model_id, attn_implementation="eager"
            )
        except TypeError:
            self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self.config = config

    def attack_image01(self, image, log_prefix=None):
        cfg = self.config
        image = image.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            clean_features, clean_scores = get_clip_feats_scores(
                self.model,
                image,
                cfg.sel_layer,
                cfg.score_method,
                cfg.attn_score_layers,
            )
            clean_features = F.normalize(
                clean_features[:, 1:].float(), dim=-1, eps=1e-12
            ).detach()
            clean_survival = budget_survival_from_scores(
                clean_scores, (cfg.k_min, cfg.k_max), cfg.score_align_eps
            )
            target_scores = target_scores_from_clean_ranks(
                ranks_from_scores(clean_scores), cfg.rank_reverse_gamma, clean_scores
            )
            target_survival = budget_survival_from_scores(
                target_scores, (cfg.k_min, cfg.k_max), cfg.score_align_eps
            )
            prior = route_prior(clean_survival, target_survival, cfg.score_align_eps)
        if cfg.random_init:
            delta = torch.empty_like(image).uniform_(-cfg.epsilon, cfg.epsilon)
            delta = project_linf_image(image, image + delta, cfg.epsilon) - image
        else:
            delta = torch.zeros_like(image)
        for step in range(cfg.steps):
            delta = delta.detach().requires_grad_(True)
            adv_image = project_linf_image(image, image + delta, cfg.epsilon)
            adv_features, adv_scores = get_clip_feats_scores(
                self.model,
                adv_image,
                cfg.sel_layer,
                cfg.score_method,
                cfg.attn_score_layers,
            )
            adv_survival = budget_survival_from_scores(
                adv_scores, (cfg.k_min, cfg.k_max), cfg.score_align_eps
            )
            route = route_loss(adv_survival, target_survival, prior, cfg.log_eps)
            tda, stats = tda_loss(
                clean_features,
                adv_features[:, 1:],
                clean_survival,
                target_survival,
                adv_survival,
                prior,
                cfg.score_align_eps,
                True,
            )
            loss = cfg.lambda_route * route + cfg.lambda_tda * tda
            gradient = torch.autograd.grad(loss, delta)[0]
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                delta = delta + cfg.alpha * gradient.sign()
                delta = project_linf_image(image, image + delta, cfg.epsilon) - image
            if log_prefix and (
                step % max(1, cfg.steps // 5) == 0 or step == cfg.steps - 1
            ):
                print(
                    "[%s] step %03d | loss %.4f route %.4f tda %.4f"
                    % (log_prefix, step, loss.item(), route.item(), tda.item())
                )
        return project_linf_image(image, image + delta, cfg.epsilon).detach()


def normalize_yes_no(answer):
    text = str(answer).lower().translate(str.maketrans("", "", string.punctuation))
    match = re.search(r"\b(yes|no)\b", text)
    return match.group(1) if match else (text.split()[0] if text.split() else "")


def ensure_yes_no_prompt(question):
    return (
        question
        if "yes or no" in question.lower()
        else question.rstrip() + " Please answer yes or no."
    )


def ensure_textvqa_prompt(question):
    return (
        question
        if "single word or a short phrase" in question.lower()
        else question.rstrip() + " " + TEXTVQA_INSTRUCTION
    )


def _answer_processor():
    global _ANSWER_PROCESSOR
    if _ANSWER_PROCESSOR is None:
        path = os.path.join(LLAVA_ROOT, "llava", "eval", "m4c_evaluator.py")
        spec = importlib.util.spec_from_file_location("cira_m4c", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ANSWER_PROCESSOR = module.EvalAIAnswerProcessor()
    return _ANSWER_PROCESSOR


def textvqa_correct(prediction, answers):
    processor = _answer_processor()
    return processor(prediction) in {processor(answer) for answer in answers}


def pct(value, total):
    return value / total * 100 if total else 0
