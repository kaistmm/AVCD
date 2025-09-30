# AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding

This is the official repository for **[Audio-Visual Contrastive Decoding (AVCD)](https://arxiv.org/abs/2505.20862)**, a simple, training-free method for mitigating hallucinations in AV-LLMs during decoding **without relying on external tools**.


---

## 🚀 Updates
- ✅ AVCD code released with forward-loop decoding  
- ✅ Accepted at **NeurIPS 2025**  

---

![Overview of AVCD](AVCD.pdf)

## 📖 Overview
- Reformulates conventional CD (Contrastive Decoding) from single-instance (e.g., video–text) to **three-modality interactions**  
- Dynamically detects the **dominant modality** and masks less dominant modalities before applying CD  
- Introduces **entropy-guided adaptive gating** to skip unnecessary forward passes and improve inference speed  

---

## ⚙️ Setup

### 1. Environment
Follow the [VideoLLaMA2 repository](https://github.com/DAMO-NLP-SG/VideoLLaMA2) setup guide (**audio-visual branch**):


### 2. Datasets

AVHBench → [GitHub](https://github.com/kaist-ami/AVHBench)
Music-AVQA → [GitHub](https://github.com/GeWu-Lab/MUSIC-AVQA)

Expected paths:

JSON configs → json/
Inference scripts → videollama2/inference/
Evaluation scripts → videollama2/eval/

### 3. Usage

```bash
git clone https://github.com/kaistmm/AVCD.git
cd Video-LLaMA2-AVCD
```

### 4. Inference

Original model
```bash
python videollama2/inference/inference_AVH_val.py
```

AVCD
```bash
python videollama2/inference/inference_AVH_val.py --use-AVCD True
```

### 4. Evaluation
This is for Accuracy (AVH1, 2, 3)
```bash
python videollama2/eval/eval_acc.py --pred-path <path_to_preds>.json
```

This is for Captioning (AVH4)
```bash
python videollama2/eval/eval_caption.py --pred-path <path_to_preds>.json --output-dir <dir>
```

This is for open-ended QA(Music-AVQA)
```bash
python videollama2/eval/eval_gpt.py --pred-path <path_to_preds>.json --output-dir <dir>
```

## 📝 Citation
```bibtex
@article{jung2025avcd,
  title={AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding},
  author={Jung, Chaeyoung and Jang, Youngjoon and Chung, Joon Son},
  journal={arXiv preprint arXiv:2505.20862},
  year={2025}
}
```# AVCD
