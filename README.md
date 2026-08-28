# CIRA

Official release code for **CIRA**, an adversarial attack against large
vision-language models with visual-token compression.

## Contents

```text
CIRA/
  run_cira_eval.py       # command-line evaluation
  cira_core.py           # CIRA objective and LLaVA/VisionZip runner
  attack_batching.py     # fixed-size attack batching
  LLaVA/                 # required LLaVA-1.5 inference code
  VisionZip/             # required VisionZip implementation
  requirements.txt
```

## Requirements

- Python 3.10
- CUDA and a matching PyTorch build
- A LLaVA-1.5 checkpoint
- A CLIP vision tower compatible with the checkpoint
- POPE, TextVQA, or MME evaluation data

Install the dependencies in a fresh environment:

```bash
conda create -n cira python=3.10 -y
conda activate cira
pip install -r requirements.txt
```

The listed PyTorch version is `2.1.2`. Install the CUDA wheel appropriate for
the target machine before installing the remaining packages when necessary.

## Data Layout

The default paths are relative to the `CIRA` directory:

```text
data/
  pope/sample_pope_1000.jsonl
  pope/sample_pope/*.jpg
  textvqa/sample_textvqa_1000.jsonl
  textvqa/sample_textvqa/*.jpg
  mme/sample_mme_1000.jsonl
  mme/sample_mme/*.jpg
```

Use `--data-root` and `--question-file` for another layout. The loader accepts
the compact JSONL files used by the release and the standard TextVQA JSON
format.

## Usage

Run a clean baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python run_cira_eval.py \
  --dataset pope \
  --attack none \
  --model-path /path/to/llava-v1.5-7b \
  --vision-tower /path/to/clip-vit-large-patch14-336 \
  --data-root /path/to/data/pope \
  --question-file /path/to/data/pope/sample_pope_1000.jsonl
```

Run CIRA and evaluate four VisionZip budgets:

```bash
CUDA_VISIBLE_DEVICES=0 python run_cira_eval.py \
  --dataset pope \
  --attack cira \
  --model-path /path/to/llava-v1.5-7b \
  --vision-tower /path/to/clip-vit-large-patch14-336 \
  --attack-model-id /path/to/clip-vit-large-patch14-336 \
  --data-root /path/to/data/pope \
  --question-file /path/to/data/pope/sample_pope_1000.jsonl \
  --budgets "[(27, 5), (54, 10), (108, 20), (162, 30)]" \
  --epsilon 4 \
  --alpha 1 \
  --steps 100
```

`--epsilon` and `--alpha` use pixel units divided by 255. Set
`--num-samples` for a short run. Results are written as JSON files under
`--output-dir`.

## Reported Metrics

- `clean_acc`: accuracy on clean images.
- `adv_acc`: accuracy after the full-token attack.
- `asr`: full-token attack success rate over clean-correct samples.
- `csfr`: compression-specific failure rate at a VisionZip budget.

## License

The CIRA-specific code is released under the project license selected by the
authors. `LLaVA/LICENSE` and `VisionZip/LICENSE` apply to the corresponding
third-party source files. Model checkpoints and benchmark data remain subject
to their original licenses and are not redistributed here.
