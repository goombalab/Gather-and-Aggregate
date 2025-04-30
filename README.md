# Gather-and-Aggregate

This repository accompanies the paper:  
**"Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism"**

We investigate the retrieval capabilities of recurrent models (like Mamba) and how they compare to Transformers. 
Our core contribution is an analysis of the *Gather-and-Aggregate (G&A)* mechanism, 
showing that retrieval behavior is driven by a small subset of heads in both Transformers and State-space models (SSM).

---

## 📂 Repository Structure

```
.
├── notebook.ipynb         # Main experiment notebook
├── kv_retrieval.py        # Tools for key-value retrieval analysis
├── utils.py               # Miscellaneous utilities
├── visualize.py           # Plotting and visualization functions
├── models/                # Architecture components (e.g., hybrid, SSM, transformer)
└── assets/                # Static images or data files
```

---

## 🚀 Quickstart

1. **Clone the repo:**
   ```bash
   git clone https://github.com/goombalab/Gather-and-Aggregate.git
   cd Gather-and-Aggregate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```

---

## 🔍 Core Experiments

- **Head Importance Analysis:**  
  Identify which attention heads perform retrieval across models like Llama and Mamba.

- **Hybrid Architectures:**  
  Replace non-retrieval heads with SSM layers to form faster, retrieval-preserving models.

- **Hidden-State Alignment:**  
  Distill representations from transformer models into hybrids to improve performance.

---

## 📄 Citation

If you find this repository useful, please cite:

```
@misc{bick2025understandingskillgaprecurrent,
      title={Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism}, 
      author={Aviv Bick and Eric Xing and Albert Gu},
      year={2025},
      eprint={2504.18574},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.18574}, 
}
```
