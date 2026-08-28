import argparse
import gc
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from attack_batching import (
    current_attack_batch_size,
    initialize_attack_batch_state,
)
from cira_core import (
    CIRAConfig,
    CIRAAttacker,
    LlavaVisionZipRunner,
    ensure_textvqa_prompt,
    ensure_yes_no_prompt,
    load_jsonl,
    normalize_yes_no,
    parse_attention_layers,
    parse_budget_pairs,
    pct,
    set_seed,
    textvqa_correct,
)


def find_pope_image_path(root, image_name):
    paths = [
        os.path.join(root, image_name),
        os.path.join(root, "sample_pope", image_name),
        os.path.join(root, "val2014", image_name),
        os.path.join(root, "images", image_name),
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[1]


def find_mme_image_path(root, image_name):
    paths = [
        os.path.join(root, image_name),
        os.path.join(root, "sample_mme", image_name),
        os.path.join(root, "images", image_name),
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[1]


def find_textvqa_image_path(root, image_id):
    image_id = os.path.splitext(os.path.basename(str(image_id)))[0]
    paths = [
        os.path.join(root, "sample_textvqa", f"{image_id}.jpg"),
        os.path.join(root, "sample_textvqa", f"{image_id}.png"),
        os.path.join(root, "sample_textvqa", f"{image_id}.jpeg"),
        os.path.join(root, "train_val_images", "train_images", f"{image_id}.jpg"),
        os.path.join(root, "train_val_images", "val_images", f"{image_id}.jpg"),
        os.path.join(root, "train_images", f"{image_id}.jpg"),
        os.path.join(root, "val_images", f"{image_id}.jpg"),
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def load_textvqa_records(path):
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)["data"]


def load_samples(args):
    if args.dataset == "pope":
        records = load_jsonl(args.question_file)
        samples = []
        for item in records:
            image_name = item["image"]
            label = normalize_yes_no(item["label"])
            samples.append(
                {
                    "id": item.get("question_id"),
                    "image_key": image_name,
                    "image_path": find_pope_image_path(args.data_root, image_name),
                    "question": ensure_yes_no_prompt(item["text"]),
                    "label": label,
                    "answers": [label],
                    "category": item.get("category", "pope"),
                }
            )
        return samples

    if args.dataset == "mme":
        records = load_jsonl(args.question_file)
        samples = []
        for item in records:
            image_name = item["image"]
            answers = item.get("answers", [])
            label = answers[0] if answers else item.get("answer", "")
            label = normalize_yes_no(label)
            samples.append(
                {
                    "id": item.get("question_id"),
                    "image_key": image_name,
                    "image_path": find_mme_image_path(args.data_root, image_name),
                    "question": item["question"],
                    "label": label,
                    "answers": [label],
                    "category": item.get("category", "unknown"),
                }
            )
        return samples

    if args.dataset == "textvqa":
        records = load_textvqa_records(args.question_file)
        samples = []
        for item in records:
            image_id = item["image_id"]
            samples.append(
                {
                    "id": item["question_id"],
                    "image_key": str(image_id),
                    "image_path": find_textvqa_image_path(args.data_root, image_id),
                    "question": ensure_textvqa_prompt(item["question"]),
                    "label": None,
                    "answers": item["answers"],
                    "category": "textvqa",
                }
            )
        return samples

    raise ValueError(f"unsupported dataset: {args.dataset}")


def is_correct(dataset, pred, sample):
    if dataset == "textvqa":
        return textvqa_correct(pred, sample["answers"])
    return normalize_yes_no(pred) == sample["label"]


def norm_pred(dataset, pred):
    if dataset == "textvqa":
        return pred
    return normalize_yes_no(pred)


def new_full_stats():
    return {"total": 0, "clean": 0, "adv": 0, "den": 0, "hit": 0}


def new_zip_stats(budgets):
    stats = {}
    for dominant, contextual in budgets:
        stats[f"zip_{dominant}_{contextual}"] = {
            "dominant": int(dominant),
            "contextual": int(contextual),
            "k": int(dominant) + int(contextual),
            "total": 0,
            "clean": 0,
            "adv": 0,
            "csfr_den": 0,
            "csfr_hit": 0,
            "den": 0,
            "hit": 0,
        }
    return stats


def avg_k_csfr(zip_stats):
    if not zip_stats:
        return 0
    return sum(pct(s["csfr_hit"], s["csfr_den"]) for s in zip_stats.values()) / len(
        zip_stats
    )


def avg_k_conditional_asr(zip_stats):
    if not zip_stats:
        return 0
    return sum(pct(s["hit"], s["den"]) for s in zip_stats.values()) / len(zip_stats)


def avg_k_asr(zip_stats):
    return avg_k_conditional_asr(zip_stats)


def update_clean_only_stats(full, zip_stats, clean_full_ok, clean_k_ok_by_key):
    full["total"] += 1
    full["clean"] += int(clean_full_ok)
    for key, clean_k_ok in clean_k_ok_by_key.items():
        zip_stats[key]["total"] += 1
        zip_stats[key]["clean"] += int(clean_k_ok)


def update_asr_stats(
    full, zip_stats, clean_full_ok, adv_full_ok, clean_k_ok_by_key, adv_k_ok_by_key
):
    full["total"] += 1
    full["clean"] += int(clean_full_ok)
    full["adv"] += int(adv_full_ok)
    full["den"] += int(clean_full_ok)
    full["hit"] += int(clean_full_ok and not adv_full_ok)
    for key in clean_k_ok_by_key:
        clean_k_ok = clean_k_ok_by_key[key]
        adv_k_ok = adv_k_ok_by_key[key]
        csfr_den = clean_full_ok and clean_k_ok
        conditional_den = csfr_den and adv_full_ok
        hit = conditional_den and not adv_k_ok
        s = zip_stats[key]
        s["total"] += 1
        s["clean"] += int(clean_k_ok)
        s["adv"] += int(adv_k_ok)
        s["csfr_den"] += int(csfr_den)
        s["csfr_hit"] += int(hit)
        s["den"] += int(conditional_den)
        s["hit"] += int(hit)
        adv_k_ok_by_key[key] = {
            "adv_correct": bool(adv_k_ok),
            "csfr_hit": bool(hit),
            "conditional_asr_hit": bool(hit),
        }


def build_metrics(full, zip_stats, attack):
    full_metrics = {
        "total": full["total"],
        "clean_acc": pct(full["clean"], full["total"]),
    }
    if attack != "none":
        full_metrics.update(
            {
                "adv_acc": pct(full["adv"], full["total"]),
                "asr_den": full["den"],
                "asr_hit": full["hit"],
                "asr": pct(full["hit"], full["den"]),
            }
        )

    zip_metrics = {}
    for key, s in zip_stats.items():
        metrics = {
            "dominant": s["dominant"],
            "contextual": s["contextual"],
            "k": s["k"],
            "total": s["total"],
            "clean_acc": pct(s["clean"], s["total"]),
        }
        if attack != "none":
            conditional_asr = pct(s["hit"], s["den"])
            metrics.update(
                {
                    "adv_acc": pct(s["adv"], s["total"]),
                    "csfr_den": s["csfr_den"],
                    "csfr_hit": s["csfr_hit"],
                    "csfr": pct(s["csfr_hit"], s["csfr_den"]),
                    "conditional_asr_den": s["den"],
                    "conditional_asr_hit": s["hit"],
                    "conditional_asr": conditional_asr,
                    "asr_den": s["den"],
                    "asr_hit": s["hit"],
                    "asr": conditional_asr,
                }
            )
        zip_metrics[key] = metrics

    avg_k = {"num_budgets": len(zip_stats)}
    if attack != "none":
        conditional_asr = avg_k_conditional_asr(zip_stats)
        avg_k.update(
            {
                "csfr": avg_k_csfr(zip_stats),
                "conditional_asr": conditional_asr,
                "asr": conditional_asr,
            }
        )
    return {
        "primary_metric": "csfr",
        "auxiliary_metric": "conditional_asr",
        "full": full_metrics,
        "zip": zip_metrics,
        "avg_k": avg_k,
    }


def print_metrics(metrics, attack):
    full = metrics["full"]
    print("\n" + "=" * 60)
    print("CIRA Evaluation Results")
    print("=" * 60)
    if attack == "none":
        print(f"full: clean={full['clean_acc']:.2f} ({full['total']} samples)")
    else:
        print(
            f"full: clean={full['clean_acc']:.2f} adv={full['adv_acc']:.2f} "
            f"asr={full['asr']:.2f} ({full['asr_hit']}/{full['asr_den']})"
        )
    for key, m in metrics["zip"].items():
        if attack == "none":
            print(f"{key}(K={m['k']}): clean={m['clean_acc']:.2f}")
        else:
            print(
                f"{key}(K={m['k']}): clean={m['clean_acc']:.2f} "
                f"adv={m['adv_acc']:.2f} "
                f"CSFR={m['csfr']:.2f} ({m['csfr_hit']}/{m['csfr_den']}) "
                f"Cond-ASR={m['conditional_asr']:.2f} "
                f"({m['conditional_asr_hit']}/{m['conditional_asr_den']})"
            )
    if attack != "none":
        print(
            f"avg_k: CSFR={metrics['avg_k']['csfr']:.2f} "
            f"Cond-ASR={metrics['avg_k']['conditional_asr']:.2f} "
            f"({metrics['avg_k']['num_budgets']} budgets)"
        )
    print("=" * 60)


def print_eval_result(args, idx, sample, clean_full_ok, adv_full_ok, full, zip_stats):
    if not args.eval_log:
        return
    if args.attack == "none":
        tqdm.write(
            f"[eval] {idx + 1:04d} id={sample['id']} image={sample['image_key']} "
            f"clean_full={int(clean_full_ok)} clean_acc={pct(full['clean'], full['total']):.1f}%"
        )
        return
    tqdm.write(
        f"[eval] {idx + 1:04d} id={sample['id']} image={sample['image_key']} "
        f"clean_full={int(clean_full_ok)} adv_full={int(adv_full_ok)} "
        f"AvgK CSFR={avg_k_csfr(zip_stats):.1f}% "
        f"AvgK Cond-ASR={avg_k_conditional_asr(zip_stats):.1f}% "
        f"Full ASR={pct(full['hit'], full['den']):.1f}%"
    )


def cuda_oom(exc):
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cublas_status_alloc_failed" in message
        or "cuda error: memory allocation" in message
    )


def release_cuda_oom_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cache_adv_tensor(adv_cache, image_key, tensor):
    adv_cache[image_key] = tensor.detach().cpu()


def cached_adv_tensor(adv_cache, image_key, device):
    return adv_cache[image_key].to(device=device)


def store_attack_chunk(chunk, args, runner, attacker, adv_cache):
    if not chunk:
        return

    effective_size = current_attack_batch_size(args)
    if len(chunk) > effective_size:
        for start in range(0, len(chunk), effective_size):
            store_attack_chunk(
                chunk[start : start + effective_size],
                args,
                runner,
                attacker,
                adv_cache,
            )
        return

    if len(chunk) == 1:
        clean_01 = chunk[0]["clean_01"]
        log_prefix = chunk[0]["image_key"] if args.attack_log else None
    else:
        clean_01 = torch.cat(
            [item["clean_01"] for item in chunk],
            dim=0,
        )
        log_prefix = None
        if args.attack_log:
            log_prefix = f"batch {chunk[0]['image_key']}.." f"{chunk[-1]['image_key']}"
    adv_01 = attacker.attack_image01(clean_01, log_prefix=log_prefix)

    for i, item in enumerate(chunk):
        adv_tensor = runner.image01_to_tensor(adv_01[i : i + 1]).detach()
        cache_adv_tensor(adv_cache, item["image_key"], adv_tensor)


def optimize_attack_batch(
    samples, start_idx, args, runner, attacker, adv_cache, clean_tensor
):
    batch_size = current_attack_batch_size(args)
    sample = samples[start_idx]

    if not args.cache_adv:
        clean_01 = runner.tensor_to_image01(clean_tensor)
        adv_01 = attacker.attack_image01(
            clean_01,
            log_prefix=(sample["image_key"] if args.attack_log else None),
        )
        adv_tensor = runner.image01_to_tensor(adv_01)
        return adv_tensor

    chunk = []
    seen = set()
    current_01 = runner.tensor_to_image01(clean_tensor)
    expected_shape = tuple(current_01.shape[1:])
    for idx in range(start_idx, len(samples)):
        item = samples[idx]
        image_key = item["image_key"]
        if image_key in adv_cache or image_key in seen:
            continue
        if not os.path.exists(item["image_path"]):
            continue

        if idx == start_idx:
            clean_01 = current_01
        else:
            image_tensor = runner.load_image_tensor(item["image_path"])
            clean_01 = runner.tensor_to_image01(image_tensor)
        if tuple(clean_01.shape[1:]) != expected_shape:
            continue

        chunk.append({"image_key": image_key, "clean_01": clean_01})
        seen.add(image_key)
        if len(chunk) >= batch_size:
            break

    store_attack_chunk(chunk, args, runner, attacker, adv_cache)
    if sample["image_key"] not in adv_cache:
        return None
    return cached_adv_tensor(
        adv_cache,
        sample["image_key"],
        runner.device,
    )


def evaluate(args):
    set_seed(args.seed)
    apply_dataset_defaults(args)
    initialize_attack_batch_state(args)
    args.attack_score_layers = parse_attention_layers(args.sel_layer)
    budgets = (
        parse_budget_pairs(args.budgets)
        if args.budgets
        else [(27, 5), (54, 10), (108, 20), (162, 30)]
    )
    samples = load_samples(args)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    runner = LlavaVisionZipRunner(
        model_path=args.model_path,
        vision_tower=args.vision_tower,
        dominant=budgets[0][0],
        contextual=budgets[0][1],
    )

    attacker = None
    if args.attack == "cira":
        attack_cfg = CIRAConfig(
            epsilon=args.epsilon / 255.0,
            alpha=args.alpha / 255.0,
            steps=args.steps,
            k_min=args.k_min,
            k_max=args.k_max,
            lambda_route=args.lambda_route,
            rank_reverse_gamma=args.rank_reverse_gamma,
            score_align_eps=args.score_align_eps,
            log_eps=args.log_eps,
            sel_layer=args.attack_score_layers[0],
            score_method=args.score_method,
            attn_score_layers=args.attack_score_layers,
            random_init=args.random_init,
        )
        attacker = CIRAAttacker(args.attack_model_id, attack_cfg, device=runner.device)

    if args.eval_log and args.attack != "none":
        print(f"[attack-batch] attack_batch_size={args.attack_batch_size}")

    full = new_full_stats()
    zip_stats = new_zip_stats(budgets)
    adv_cache = {}
    results = []
    errors = defaultdict(int)
    missing = []

    progress = tqdm(samples, desc=f"Online {args.dataset.upper()}", total=len(samples))
    for idx, sample in enumerate(progress):
        try:
            if not os.path.exists(sample["image_path"]):
                missing.append(sample["image_path"])
                errors["missing_image"] += 1
                continue

            clean_tensor = runner.load_image_tensor(sample["image_path"])
            adv_tensor = None
            if args.attack == "cira":
                if args.cache_adv and sample["image_key"] in adv_cache:
                    adv_tensor = cached_adv_tensor(
                        adv_cache,
                        sample["image_key"],
                        runner.device,
                    )
                else:
                    adv_tensor = optimize_attack_batch(
                        samples,
                        idx,
                        args,
                        runner,
                        attacker,
                        adv_cache,
                        clean_tensor,
                    )

            question = sample["question"]
            clean_full_pred = runner.predict_tensor(
                clean_tensor, question, budget=None, max_tokens=args.max_tokens
            )
            clean_full_ok = is_correct(args.dataset, clean_full_pred, sample)

            adv_full_pred = None
            adv_full_ok = False
            if adv_tensor is not None:
                adv_full_pred = runner.predict_tensor(
                    adv_tensor, question, budget=None, max_tokens=args.max_tokens
                )
                adv_full_ok = is_correct(args.dataset, adv_full_pred, sample)

            clean_k_ok_by_key = {}
            adv_k_ok_by_key = {}
            zip_rows = {}
            for budget in budgets:
                key = f"zip_{budget[0]}_{budget[1]}"
                clean_k_pred = runner.predict_tensor(
                    clean_tensor, question, budget=budget, max_tokens=args.max_tokens
                )
                clean_k_ok = is_correct(args.dataset, clean_k_pred, sample)
                clean_k_ok_by_key[key] = clean_k_ok
                row = {
                    "clean": norm_pred(args.dataset, clean_k_pred),
                    "clean_correct": clean_k_ok,
                }
                if adv_tensor is not None:
                    adv_k_pred = runner.predict_tensor(
                        adv_tensor, question, budget=budget, max_tokens=args.max_tokens
                    )
                    adv_k_ok = is_correct(args.dataset, adv_k_pred, sample)
                    adv_k_ok_by_key[key] = adv_k_ok
                    row.update(
                        {
                            "adv": norm_pred(args.dataset, adv_k_pred),
                            "adv_correct": adv_k_ok,
                        }
                    )
                zip_rows[key] = row

            if args.attack == "none":
                update_clean_only_stats(
                    full, zip_stats, clean_full_ok, clean_k_ok_by_key
                )
            else:
                update_asr_stats(
                    full,
                    zip_stats,
                    clean_full_ok,
                    adv_full_ok,
                    clean_k_ok_by_key,
                    adv_k_ok_by_key,
                )
                for key, value in adv_k_ok_by_key.items():
                    if isinstance(value, dict):
                        zip_rows[key].update(
                            {
                                "csfr_hit": value["csfr_hit"],
                                "conditional_asr_hit": value["conditional_asr_hit"],
                                "hit": value["conditional_asr_hit"],
                            }
                        )
            results.append(
                {
                    "sample_index": idx,
                    "id": sample["id"],
                    "image_key": sample["image_key"],
                    "image_path": sample["image_path"],
                    "question": question,
                    "label": sample["label"],
                    "answers": sample["answers"],
                    "category": sample["category"],
                    "clean_full": norm_pred(args.dataset, clean_full_pred),
                    "clean_full_correct": clean_full_ok,
                    "adv_full": (
                        norm_pred(args.dataset, adv_full_pred)
                        if adv_full_pred is not None
                        else None
                    ),
                    "adv_full_correct": adv_full_ok if adv_tensor is not None else None,
                    "full_hit": (
                        bool(clean_full_ok and not adv_full_ok)
                        if adv_tensor is not None
                        else None
                    ),
                    "zip": zip_rows,
                }
            )
            print_eval_result(
                args, idx, sample, clean_full_ok, adv_full_ok, full, zip_stats
            )

            postfix = {"clean": f"{pct(full['clean'], full['total']):.1f}%"}
            if args.attack != "none":
                postfix.update(
                    {
                        "AvgK CSFR": f"{avg_k_csfr(zip_stats):.1f}%",
                        "AvgK Cond-ASR": f"{avg_k_conditional_asr(zip_stats):.1f}%",
                        "Full ASR": f"{pct(full['hit'], full['den']):.1f}%",
                    }
                )
            progress.set_postfix(postfix)
        except Exception as exc:
            if cuda_oom(exc):
                release_cuda_oom_memory()
                print(
                    "\n[fatal] CUDA OOM with fixed attack batch "
                    f"{current_attack_batch_size(args)}; automatic "
                    "batch-size backoff is disabled."
                )
                raise
            errors[str(type(exc).__name__)] += 1
            print(f"\nError processing {sample.get('id')}: {exc}")

    metrics = build_metrics(full, zip_stats, args.attack)
    print_metrics(metrics, args.attack)
    if errors:
        print("\nErrors encountered:")
        for key, value in errors.items():
            print(f"  {key}: {value}")
    if missing:
        print(f"\nMissing images: {len(missing)}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"{args.dataset}_online_{args.attack}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "budgets": budgets,
                "metrics": metrics,
                "errors": dict(errors),
                "missing_images": missing,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIRA evaluation with LLaVA and VisionZip"
    )
    parser.add_argument(
        "--dataset", type=str, default="pope", choices=["pope", "textvqa", "mme"]
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vision-tower", type=str, required=True)
    parser.add_argument("--attack-model-id", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--attack", type=str, default="cira", choices=["none", "cira"])
    parser.add_argument(
        "--budgets", type=str, default="[(27, 5), (54, 10), (108, 20), (162, 30)]"
    )
    parser.add_argument(
        "--score-method", type=str, default="attn", choices=["norm", "attn"]
    )
    parser.add_argument("--sel-layer", type=str, default="-2")
    parser.add_argument("--k-min", type=int, default=32)
    parser.add_argument("--k-max", type=int, default=192)
    parser.add_argument("--lambda-route", type=float, default=1.0)
    parser.add_argument("--rank-reverse-gamma", type=float, default=1.0)
    parser.add_argument("--score-align-eps", type=float, default=1e-6)
    parser.add_argument("--log-eps", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--attack-batch-size",
        type=int,
        default=2,
        help=(
            "Fixed maximum attack-optimization batch size. Images with "
            "different processed shapes may form smaller batches."
        ),
    )
    parser.add_argument("--random-init", action="store_true", default=False)
    parser.add_argument(
        "--no-cache-adv", dest="cache_adv", action="store_false", default=True
    )
    parser.add_argument(
        "--no-attack-log", dest="attack_log", action="store_false", default=True
    )
    parser.add_argument(
        "--no-eval-log", dest="eval_log", action="store_false", default=True
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_dataset_defaults(args):
    defaults = {
        "pope": ("./data/pope", "./data/pope/sample_pope_1000.jsonl"),
        "textvqa": (
            "./data/textvqa",
            "./data/textvqa/sample_textvqa_1000.jsonl",
        ),
        "mme": ("./data/mme", "./data/mme/sample_mme_1000.jsonl"),
    }
    default_root, default_questions = defaults[args.dataset]
    if args.data_root is None:
        args.data_root = default_root
    if args.question_file is None:
        args.question_file = default_questions
    if args.attack_model_id is None:
        args.attack_model_id = args.vision_tower


if __name__ == "__main__":
    evaluate(parse_args())
