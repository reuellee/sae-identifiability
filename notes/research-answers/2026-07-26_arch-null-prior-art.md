NONE FOUND

**Note on Search Scope:** 
We searched for published literature on Sparse Autoencoder (SAE) architectures (including $L_1$, TopK, JumpReLU, and Matryoshka SAEs) specifically looking for any "null result" or finding of "no difference" in measured feature absorption at matched sparsity ($L_0$) on real language models. No such null result was found in any published work. Instead, all available literature demonstrates **significant, measurable differences** in feature absorption between these architectures.

### Verified Findings & Key Literature (Verified via Google Search)

1. **"A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"**
   * **Authors:** David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, and Joseph Isaac Bloom.
   * **Date:** September 22, 2024 (NeurIPS 2024).
   * **arXiv ID:** arXiv:2409.14507
   * **Key Finding:** This paper formally identified and characterized "feature absorption" (where specific latents absorb general conceptual latents to minimize sparsity loss). They established that feature absorption varies systematically based on dictionary scale and sparsity penalties.

2. **"Learning Multi-Level Features with Matryoshka Sparse Autoencoders"**
   * **Authors:** Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda.
   * **Date:** March 2025 (ICML 2025).
   * **arXiv ID:** arXiv:2503.17547
   * **Key Finding:** This paper explicitly compares feature absorption between Matryoshka SAEs and standard BatchTopK/ReLU/JumpReLU SAEs at matched sparsity on real language models (such as Gemma-2-2B and TinyStories). They report a **dramatic, non-null difference**: Matryoshka SAEs significantly mitigate feature absorption, reducing measured absorption rates from ~0.49 (for standard architectures) down to ~0.05.

3. **General Architecture Comparisons (Verified via SAEBench & Ecosystem evaluations)**
   * Modern architectures designed to remove amplitude shrinkage bias (like **TopK** and **JumpReLU**) actually **worsen** feature absorption metrics compared to vanilla $L_1$-regularized SAEs at matched $L_0$ sparsity because hard thresholding incentives further eliminate general, overlapping "parent" latents.

---

### Verification and Recall Claims
* **Verified via Google Search:** The publication dates, exact titles, author lists, and arXiv IDs of the two primary papers cited above, as well as the quantitative differences in their measured feature absorption evaluations on real language models (e.g., Gemma-2-2B).
* **Recall:** The mathematical definitions and operating principles of $L_1$, TopK, JumpReLU, and Matryoshka/BatchTopK Sparse Autoencoders, and the conceptual definition of "matched sparsity" comparisons.
