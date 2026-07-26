**NONE FOUND**

### Search Summary & Scope
We performed targeted queries across Google, LessWrong, and the AI Alignment Forum using terms such as:
* `"absorbing parent" AND "absorbed child" AND "latent" OR "feature"`
* `"child feature's mass" OR "child feature mass" sparse autoencoder OR "SAE"`
* `"absence of a firing child latent" OR "absence of a firing child"`
* `"feature absorption" AND ("positive evidence" OR "mass" OR "evidence") AND ("parent" OR "child")`

None of these searches returned any papers, preprints, or forum posts requiring "positive evidence" of a parent latent absorbing a child latent's mass (as opposed to inferring absorption from the absence of a firing child latent).

---

### Verified Claims vs. Recall

**Verified via Search:**
1. **Seminal Work:** The primary framework for studying feature absorption was established in the paper:
   * **Title:** *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*
   * **Authors:** David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, and Joseph Bloom
   * **Date:** September 2024 (arXiv:2409.13575)
2. **Direction of Absorption:** In Chanin et al. (2024) and subsequent discussions (e.g., "Tying the SAE encoder and decoder weights solves feature absorption" by Joseph Bloom, October 2024), feature absorption is defined in the **opposite direction**: a specific *child* latent (e.g., "elephant") absorbs the direction of a general *parent* feature (e.g., "starts with E"). 
3. **Standard Metric:** Because the specific child latent absorbs the general parent direction, absorption is measured by the **absence** of the parent latent's activation when the child feature is present (i.e., detecting "holes" in the parent's activation map). 

**Recall:**
1. In the reverse direction—where child concepts are absorbed by a parent latent—there is no established benchmark or paper in the literature requiring researchers to positively isolate and map the child's activation mass within a decoded parent latent to prove absorption. Under standard SAE mechanics, when a child concept fails to activate its own specialized latent, the information is implicitly assumed to be handled by the active parent latent (or distributed elsewhere), but this is not measured as a formal "positive evidence" constraint.
