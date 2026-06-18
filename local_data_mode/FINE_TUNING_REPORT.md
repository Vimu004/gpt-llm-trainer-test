# Fine-Tuning Report: Structured Clinical SOAP Note Generator

This report documents the design, implementation, and successful training of a task-specific **Structured Clinical SOAP Note Generator** using a quantized LLaMA-2 variant (**TinyLlama-1.1B**). 

---

## 1. Project Overview & Use Case

* **Objective**: Automatically transcribe raw, unstructured conversational transcripts between doctors and patients into highly professional, structured clinical SOAP notes (Subjective, Objective, Assessment, Plan).
* **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (selected for compatibility with standard Colab GPU limits).
* **Fine-Tuning Method**: QLoRA (4-bit quantization with Parameter-Efficient Fine-Tuning).
* **Workflow Architecture**: **Hybrid Execution Pipeline**
  * **Phase 1 (Local Mac)**: Synthetic dataset generation using Azure OpenAI (`gpt-4o`). This saved Colab compute hours and bypassed runtime disconnections.
  * **Phase 2 (Google Colab)**: GPU-accelerated model training using the generated datasets (`train.jsonl` and `test.jsonl`) mounted directly from Google Drive.

---

## 2. Key Challenges & Technical Resolutions

During execution, several system errors were encountered and resolved to ensure complete stability:

1. **PyArrow Binary Mismatches**:
   * *Issue*: `pyarrow.lib.IpcReadOptions size changed, indicating binary incompatibility`.
   * *Fix*: Implemented a self-healing package installer block at the very top of the notebook that automatically checks and aligns `pyarrow` and `datasets` binaries, cleanly restarting the container if a mismatch is detected.
2. **Modern TRL API Compatibility**:
   * *Issue*: `TypeError: SFTTrainer got unexpected keyword arguments` (like `max_seq_length` and `dataset_text_field`).
   * *Fix*: Migrated the training arguments from `TrainingArguments` to **`SFTConfig`** and renamed the trainer's `tokenizer` parameter to `processing_class` to match modern Hugging Face standards.
3. **BFloat16 Precision Alignment**:
   * *Issue*: `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`.
   * *Fix*: Disabled the float16 mixed-precision `GradScaler` by setting `fp16=False` and `bf16=True`, and loaded the model consistently using `torch.bfloat16` to match the model's native dtype.
4. **Positional Embedding Overflow**:
   * *Issue*: `Token indices sequence length is longer than the specified maximum sequence length for this model (5200 > 2048)`. Extremely long input prompts caused the model to output repetitive gibberish.
   * *Fix*: Added a preprocessing step that truncates the transcript input to **600 tokens** during mapping and inference. This ensures the entire context (prompt + response) stays safely within the model's 2,048-token context window.

---

## 3. Training Progress & Metrics

* **Epochs**: 10
* **Gradient Accumulation Steps**: 2
* **Total Steps**: 110 steps
* **Training Time**: ~25 minutes (1,504 seconds on T4 GPU)
* **Final Training Loss**: **0.5953** (reduced from `1.765`)
* **Final Validation Loss**: **1.176** (reduced from `1.662`)
* **Mean Token Accuracy**: **71.26%**

---

## 4. Verification Output

When tested on a validation transcript, the model successfully formatted the output into the structured clinical SOAP format:

```text
Subjective:
- Chief Complaint: Shortness of Breath; Deterioration over several weeks; Progressively Worsening Over Last Week; Feeling "Drowned"; Leg Swelling (Ankles); Chest Pain...
- History of Present Illness:
  - 68-year-old male with long-standing type II diabetes mellitus, CKD stages 3–4, HFrEF (EF ~30%), prior CAD, and history of hyperglycemic crisis, presents with acute dyspnea...
```

---

## 5. Visual Evidence & Screenshots

*Capture and insert the following screenshots from your Google Colab and Google Drive sessions into this report to complete your submission:*

### Screenshot 1: Successful QLoRA Training Progress
> **Instructions**: Take a screenshot of the training execution logs in Colab showing the steps, training losses decreasing, and the final line `'train_loss': '1.101', 'epoch': '10'`.
>
> 📸 *[Insert Training Progress Screenshot Here]*

### Screenshot 2: Clinical SOAP Note Generation (Inference)
> **Instructions**: Take a screenshot of the final test cell output showing the prompt ending in `[/INST]` followed by the generated structured SOAP note beginning with `Subjective:`.
>
> 📸 *[Insert Inference Output Screenshot Here]*

### Screenshot 3: Model Saved to Google Drive
> **Instructions**: Open Google Drive in your browser, navigate to your `MyDrive/llama-2-7b-custom/` folder, and take a screenshot showing the saved files (`adapter_config.json`, `adapter_model.safetensors`, `special_tokens_map.json`, etc.).
>
> 📸 *[Insert Google Drive Saved Files Screenshot Here]*
