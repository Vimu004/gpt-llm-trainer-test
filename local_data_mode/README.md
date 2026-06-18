# Local Data Generation & Colab Fine-Tuning Guide

This folder contains a workspace package that allows you to run the data generation phase **locally on your computer** (using your local python environment and `.env` credentials) to save time, prevent Google Colab timeouts, and ensure complete stability. Once generated, you upload the datasets to Google Drive, and the notebook will train the model instantly on Colab's GPU.

---

## Step 1: Generate Data Locally

1. Open your terminal and navigate to this directory:
   ```bash
   cd local_data_mode
   ```
2. Install the lightweight local requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the generator script:
   ```bash
   python generate_data.py
   ```
   * **Note**: This script reads your credentials directly from the `.env` file in the parent folder.
   * **Checkpoints**: It saves progress to `checkpoint_examples.json` after every single generation. If it is interrupted, running the command again will resume right where you left off.
   * **Output**: When complete, it produces `train.jsonl` and `test.jsonl` in this directory.

---

## Step 2: Upload Files to Google Drive

1. Open **Google Drive** in your browser.
2. Create a folder in your **MyDrive** root named:
   `gpt-llm-trainer`
3. Upload `train.jsonl` and `test.jsonl` from this folder directly into your new `gpt-llm-trainer` Google Drive folder.

---

## Step 3: Run Fine-Tuning in Google Colab

1. Upload the `One_Prompt___Fine_Tuned_LLaMA_2_18_june.ipynb` notebook from this directory to Google Colab.
2. In Google Colab, select **Runtime** > **Change runtime type** and ensure you have selected **T4 GPU** or higher.
3. Click **Run All** (or run the cells sequentially).
4. Since you generated the data locally, **this notebook is a clean, dedicated training-only notebook**. It contains zero OpenAI API calls, meaning:
   * You do **not** need to add your Azure OpenAI keys or secrets to Google Colab.
   * There are no slow LLM calls on Colab; it mounts your Google Drive, loads `train.jsonl` and `test.jsonl` directly, and starts training immediately!
