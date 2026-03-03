# Copilot Instructions for EasyEdit Workspace

The EasyEdit repository is a research codebase for **knowledge editing** and **real-time steering**
of large language models (LLMs).  It lives under `/home/student/arthas/EasyEdit3` and splits into
two loosely coupled Python packages:

* `easyeditor/` – the original editing framework with algorithms such as ROME, MEMIT, GRACE, LoRA, …
* `steer/` – the “EasyEdit‑2” steering toolkit; it mirrors the editing API (vectors, appliers, …
  and its own alg_dict) and has a few standalone drivers.

These instructions are written for an AI coding agent; they highlight the things you need to know
to make edits, add features, or debug in this repository.

---

## Architecture overview

* **Core packages**
  * `easyeditor/dataset` – classes like `ZsreDataset`, `CounterFactDataset`, `CaptionDataset`,
    `SafetyDataset`.  They load JSON/JSONL, tokenize prompts, and produce request batches for
    `BaseEditor`.
  * `easyeditor/models` – one sub‑package per method.  Each defines a `*HyperParams` dataclass and
    an `apply_*_to_model` function or `RewriteExecutor` which mutates a HuggingFace model in‑place.
    Examples: `rome/`, `memit/`, `lora/`, `wise/`, `alphaedit/`, `spphere/`, etc.
  * `easyeditor/editors` – editor abstractions.  `BaseEditor` (in `editor.py`) handles tokenization,
    batching, evaluation and logging; there are subclasses or wrappers for steering,
    multimodal, safety, concept editing, per‑example editing, etc.
  * `easyeditor/evaluate` – evaluation helpers (`compute_edit_quality`,
    `test_generation_quality`), generation wrappers, and the `run_LLM_evaluation.py` script.
  * `easyeditor/util` – misc helpers (`alg_dict.py`, `globals.py`, `nethook.py`, `hparams.py`,
    `HyperParams` wrapper).

* **Registry pattern**
  * `easyeditor/util/alg_dict.py` contains `ALG_DICT` (string→(hyperparams_cls,apply_fn)),
    `ALG_MULTIMODAL_DICT`, `PER_ALG_DICT`.  New methods must register here and supply a
    hyperparams dataclass and optionally a dataset class.
  * Steering has a parallel registry at `steer/utils/alg_dict.py`.

* **Hyperparameters**
  * Stored as JSON/YAML under `hparams/<METHOD>/<model‑name>/*.yaml` or in a Hydra config
    (`hparams/Steer/config.yaml`).  They are loaded via `HyperParams.from_json()` or Hydra.
  * Float values may be strings; see `HyperParams.construct_float_from_scientific_notation`.

* **Top‑level drivers & examples**
  * Root contains dozens of throwaway scripts (`train_dental_lora.py`,
    `deploy_dental_robot*.py`, `autoTest*.py`, …) used for domain‑specific experiments.
  * The canonical examples live in `examples/`; each `run_*.py` follows a pattern: parse args,
    load hyperparams/dataset/type from `alg_dict`, instantiate an editor or trainer, then call
    `editor.edit()` or training loop.  Use `python -m` when relative imports are needed.
  * Notebooks in `tutorial-notebooks/` provide interactive demos.
  * Domain‑specific tools such as the dental QA pipeline live at the workspace root and rely on
    the `data/` folder.

* **Data**
  * Raw datasets and conversion scripts are in `data/`; edited datasets (JSONL) are loaded by
    dataset classes.  Dental files like `easyedit_dental_qa.jsonl` are examples.
  * Conversion scripts (`convert_dental_data.py`, `convertCmexam.py`) transform source formats.

* **Deployment & evaluation**
  * Deployment to vLLM servers is handled by `deploy_dental_robot*.py`; these define
    prompt‑construction helpers (`build_qa_prompt`, `build_choice_prompt`) and use vLLM to serve
    LoRA‑adapted Qwen models.
  * Evaluation scripts (`autoTest*.py`) loop over datasets and extract answers by regex.

---

## Developer workflows

1. **Environment setup**
   ```bash
   pip install -r requirements.txt      # core deps
   pip install -e .                     # editable install
   # larger‑model experiments:
   pip install -r requirements25.txt
   ```
   GPU users often install `torch`, `transformers`, `accelerate`, `bitsandbytes` and `peft`
   separately; see `colab_requirements.txt`.

2. **Running an edit or training**
   * Copy an example in `examples/` or one of the root scripts.
   * Supply `--editing_method`, `--hparams_dir`, `--data_dir`, etc.
   * Set `sequential_edit=True` for stateful (WISE‑style) runs.
   * Use `model_parallel=True` in hyperparams or `accelerate` flags for multi‑GPU.

3. **Adding a new algorithm**
   * Add package under `easyeditor/models/<name>/`, include `HyperParams` and an `apply`/executor.
   * Register in `easyeditor/util/alg_dict.py` (and steering’s registry if needed).
   * Add YAML hyperparams under `hparams/<name>/…` and a simple example in `examples/`.
   * Add dataset class if you need a new input schema.

4. **Steering vectors**
   * See `steer/vector_generators` and `steer/vector_appliers`; add new ones as needed.
   * YAML/Hydra config in `hparams/Steer/config.yaml` controls behavior.
   * Demo app in `demo/EasySteer_demo/app.py` shows end‑to‑end usage.

5. **Logging & debugging**
   * `make_logs()` is called by editors; logs live in `logs/run.log` and also print to stdout.
   * `easyeditor/util/nethook` lets you inspect activations/gradients.
   * Watch for tokenizer assertions (`tok.padding_side` in `BaseEditor.__init__`).
   * Use `python -m easyeditor.util.nethook` for interactive exploration.

6. **Testing**
   * There is not a unified test suite – most “tests” are the example scripts and notebooks.
   * A small core of unit tests lives under `easyeditor/models/melo/peft_egg` for PEFT code.
   * Run `pytest` in that folder to exercise them.

7. **Miscellaneous**
   * `setup.py` is minimal; installing in editable mode is the norm.
   * Dockerfile exists for container builds but isn’t used by examples.
   * Many root scripts are quick hacks; read the first 20 lines to understand the CLI.

---

## Conventions & patterns

* Names in registries are case‑sensitive strings.  Changing a name requires edits to the YAML
  directories and any command‑line invocations.
* Hyperparams objects mirror JSON fields exactly; unknown keys raise errors.
* Models are loaded via `transformers` with `trust_remote_code=True` for Qwen families.
* 4‑bit quantization (`QLoRA`, `bitsandbytes`) gets special handling in
  `BaseEditor.__init__`.
* Output metrics default to `./<alg>_results.json` unless `metrics_save_dir` is specified.
* Training scripts often set `batch_size=1` and use `gradient_accumulation_steps` to emulate
  single‑edit semantics (see dental LoRA scripts).
* Sequential edits keep model state across calls; non‑sequential reloads the base model each
  example.
* `data/portability/One Hop/...` is the layout expected by some benchmark scripts (see
  `examples/README.md`).

---

## Domain experiments (dental QA)

A large chunk of the workspace is devoted to Chinese dental QA experiments with Qwen models:

* Training: `train_dental_lora*.py` (1.5B→25B variants).
* Deployment: `deploy_dental_robot.py`, `deploy_dental_robot25.py` (vLLM serving).
* Testing: `autoTest*.py` (deterministic inference, answer extraction).
* Data: dental JSONL files in `data/`; conversion tools in the same folder.
* Model artifacts: `dental_qwen2.5_<size>_lora/` directories with safetensors and configs.

Workflow:
```sh
python train_dental_lora.py
python deploy_dental_robot.py
python autoTest7.py
```

---

## External dependencies

* Hugging Face Transformers, tokenizers, Accelerate, bitsandbytes, PEFT, vLLM.
* Pre‑computed momentum stats for MEMIT/ROME (Wikipedia `.npz` files).
* Flask for the demo apps (`demo/*_demo/app.py`).
* Dental experiments rely on custom regex parsing but otherwise standard libraries.

---

> 📝 **Tip for AI agents:** search with `grep`/`semantic_search` for `ALG_DICT`, `BaseEditor`,
> or the specific algorithm name when you need examples.  If you’re adding code, follow the
> existing naming patterns and update the registry; new hyperparams files are trivial JSON.

Please review and suggest any missing pieces.
