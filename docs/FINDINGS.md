# Findings

Measured results from this repository. Every number below comes from a file in the repository, named under each finding.

## Does a small LoRA adapter change what Qwen2.5-0.5B answers?

**How it was tested:** Qwen2.5-0.5B-Instruct was fine-tuned with LoRA (rank 16, alpha 32, dropout 0.05, on the q, k, v and o
projections) on a 600-example Databricks Dolly-15k subset (540 train, 60 held out) on a free Colab T4 GPU
(`finetuning/train_lora.py`). The base and fine-tuned models then answered 5 held-out instructions with greedy decoding
(`finetuning/compare.py`).

| Result | Value |
|---|---|
| Training loss, epochs 1 / 2 / 3 | 1.779 / 1.760 / 1.650 |
| Held-out answers changed by the adapter | 1 of 5 |
| Held-out answers identical to the base model | 4 of 5 |

The changed answer is the open-ended "fun things to do in New York City" prompt; the four short factual prompts give the same text
as the base model.

**Decision:** the result is reported as it is: a small adapter on 540 examples changes open-ended answers, not short factual ones.
The test (`tests/test_finetuned_adapter.py`) checks that the adapter loads, has real weights and changes the output on at least one
prompt, rather than claiming a quality gain.

**Skills shown:** LoRA with PEFT, Hugging Face Transformers, Qwen2.5 instruction tuning, greedy-decoding comparison, Colab GPU training,
unittest.

**Source:** `finetuning/comparison_results.json` (5 base and fine-tuned answers), `finetuning/adapter/adapter_config.json`,
`tests/test_finetuned_adapter.py`; the per-epoch loss is stated in `README.md` (the Colab training log is not stored in the repository).
