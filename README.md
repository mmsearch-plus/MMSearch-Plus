# [ICLR2026] MMSearch-Plus✨: Benchmarking Provenance-Aware Search for Multimodal Browsing Agents

Official repository for the paper "[MMSearch-Plus: Benchmarking Provenance-Aware Search for Multimodal Browsing Agents](https://arxiv.org/abs/2508.21475)".

🌟 For more details, please refer to the project page with examples at [https://mmsearch-plus.github.io/](https://mmsearch-plus.github.io).


[[🌐 Webpage](https://mmsearch-plus.github.io/)] [[📖 Paper](https://arxiv.org/pdf/2508.21475)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/Cie1/MMSearch-Plus)] [[🏆 Leaderboard](https://mmsearch-plus.github.io/#leaderboard)]


## 💥 News

- **[2026.02.04]** 🌟 MMSearch-Plus is now in [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). There is a simple no-search implementation [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/mmsearch_plus).
- **[2026.01.29]** 🌟 Thanks [Vision-DeepResearch](https://osilly.github.io/Vision-DeepResearch/) for evaluating on our benchmark. It achieved a new SOTA of 37.8, surpassing the previous best of 37.6 from o3.
- **[2025.01.26]** 🎊 MMSearch-Plus is accepted by ICLR 2026!
- **[2025.09.26]** 🔥 We update the [arXiv paper](https://arxiv.org/abs/2508.21475) and release all MMSearch-Plus data samples in [huggingface dataset](https://huggingface.co/datasets/Cie1/MMSearch-Plus).
- **[2025.08.29]** 🚀 We release the [arXiv paper](https://arxiv.org/abs/2508.21475).

## 📌 ToDo

- Agentic rollout framework code
- Evaluation script
- Set-of-Mark annotations

## Usage

### Dataset Usage

Load the dataset with automatic decryption using your canary string:

```python
from decrypt_after_load import decrypt_mmsearch_plus
    
# Load and decrypt in one step
decrypted_ds = decrypt_mmsearch_plus(
    dataset_path="Cie1/MMSearch-Plus",
    canary='your_canary_string' # Set the canary string (hint: it's the name of this repo without username)
)
```

## 👀 About MMSearch-Plus

MMSearch-Plus is a challenging benchmark designed to test multimodal browsing agents' ability to perform genuine visual reasoning. Unlike existing benchmarks where many tasks can be solved with text-only approaches, MMSearch-Plus requires models to extract and use fine-grained visual cues through iterative image-text retrieval.

### Key Features

🔍 **Genuine Multimodal Reasoning**: 311 carefully curated tasks that cannot be solved without visual understanding

🎯 **Fine-grained Visual Analysis**: Questions require extracting spatial cues and temporal traces from images to find out-of-image facts like events, dates, and venues

🛠️ **Agent Framework**: Model-agnostic web agent with standard browsing tools (text search, image search, zoom-in)

📍 **Set-of-Mark (SoM) Module**: Enables provenance-aware cropping and targeted searches with human-verified bounding box annotations

### Dataset Structure

Each sample contains:
- Quuestion text and images
- Ground truth answers and alternative valid responses
- Metadata including arXiv id (if an event is a paper), video URL (if an event is a video), area and subfield

### Performance Results

Evaluation of closed- and open-source MLLMs shows:
- Best accuracy is achieved by o3 with full rollout: **36.0%** (indicating significant room for improvement)
- SoM integration provides consistent gains up to **+3.9 points**
- Models struggle with multi-step visual reasoning and cross-modal information integration 

<p align="center">
    <img src="https://raw.githubusercontent.com/mmsearch-plus/mmsearch-plus.github.io/main/static/images/teaser.png" width="80%"> <br>
    The overview of three paradigms for multimodal browsing tasks that demand fine-grained visual reasoning.
</p>



<p align="center">
    <img src="https://raw.githubusercontent.com/mmsearch-plus/mmsearch-plus.github.io/main/static/images/real-teaser.jpg" width="80%"> <br>
    The overview of an example trajectory for a task in <b>MMSearch-Plus</b>.
</p>

## 🏆 Leaderboard

### Contributing to the Leaderboard

🚨 The [Leaderboard](https://mmsearch-plus.github.io/#leaderboard) is continuously being updated, welcoming the contribution of your excellent LMMs!

## 🔖 Citation

If you find **MMSearch-Plus** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{tao2025mmsearch,
  title={MMSearch-Plus: A Simple Yet Challenging Benchmark for Multimodal Browsing Agents},
  author={Tao, Xijia and Teng, Yihua and Su, Xinxing and Fu, Xinyu and Wu, Jihao and Tao, Chaofan and Liu, Ziru and Bai, Haoli and Liu, Rui and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2508.21475},
  year={2025}
}
```
