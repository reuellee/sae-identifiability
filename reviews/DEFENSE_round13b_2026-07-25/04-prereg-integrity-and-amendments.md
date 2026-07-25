# Defense round13b — Q4: prereg-integrity-and-amendments

_Examiner: Gemini 2.5 Pro via gemx. Generated 2026-07-25T16:38:17Z. Single-shot._

# THESIS DEFENSE INTERROGATION: PRE-REGISTRATION INTEGRITY & AMENDMENTS

**Examiner:** Hostile External Examiner  
**Focus Area:** Protocol Integrity, Git Chronology, and Degrees of Freedom in Round 13b  

Candidate, you have designed a self-contained "pre-registration and amendment" process that has the *aesthetic* of open science, but possesses the cryptographic and structural security of a wet paper bag. Under hostile auditing, your claim of pre-results protocol-binding is entirely unverified.

---

### PART I: THE CHRONOLOGICAL VERIFICATION FAILURE (THE "RETROACTIVE" LOOPHOLE)

#### Finding 1: Local Git History is Chronologically Non-Binding
*   **The Claim:**
    > "Status: LOCKED before any 13b SAE is trained. Lock = the commit adding this file. ... Amendments (2026-07-25, both PRE-RESULTS — declared before any 13b SAE was trained)" (`notes/prereg-round13b-capacity.md`)
*   **Why it fails / is unsupported:**
    A git commit is a purely local database. The timestamps associated with any commit (`GIT_AUTHOR_DATE` and `GIT_COMMITTING_DATE`) are simple text fields in the commit object. Any developer can forge them to any arbitrary date in the past using:
    ```bash
    git commit --date="2026-07-25 10:00:00" -m "Lock round13b"
    ```
    Even if you push to GitHub, a force-push (`git push --force`) can overwrite branches and manipulate public history. Your claim that "the evaluator was locked before training" is an honor-system self-report. A motivated author could easily run the entire 48-SAE grid, observe that $m=8192$ breaks monotonicity or that a fixed calibration grid fails, draft the "amendments" post-hoc to retroactively justify dropping the cell and changing the calibration logic, commit them with a back-dated timestamp, and force-push the branch.
*   **Specific Evidence / Experiment to settle it:**
    Local repository history is useless. To verify chronological locking, you must provide a **public, third-party immutable cryptographic timestamp** for the commit hash (`sha1`) of both the preregistered file and the evaluator script. This requires querying public API logs (e.g., GitHub's event API, public OSF registration timelines, or an on-chain transaction) to prove that the exact commit hash containing the amendments was indexed *before* the creation timestamps of the training run artifacts on Google Cloud Storage.
*   **Severity:** **MAJOR**
*   **Classification:** `[PRE-RESULTS-OK]` (This requires a change in protocol reporting and the provision of external logs without retraining).

---

### PART II: THE "L0-BLIND" CALIBRATION FALLACY

#### Finding 2: Adaptive L0-Calibration is Causal to the Absorption Endpoint
*   **The Claim:**
    > "Outcome-blind: calibration reads **only** the reported held-out L0 and never sees any absorption quantity — as the original design already stated. The target (L0 closest to 32...) is unchanged; only the search over λ changes." (`notes/prereg-round13b-capacity.md`)
*   **Why it fails / is unsupported:**
    Your claim of "outcome blindness" relies on a naive separation of $L_0$ (sparsity) and your absorption metric. They are not independent. Your Round 13a post-hoc analysis explicitly proved that:
    1. The single-latent absorption endpoint has $R^2 = 0.673$ with the main latent's selectivity.
    2. The family-corrected endpoint (`rate_fam`) still has $R^2 = 0.381$ with the maximum selectivity of the family.
    3. Selectivity is heavily driven by threshold suppression—which is *directly* controlled by the sparsity penalty $\lambda$.

    By changing the calibration mechanism from a fixed grid to an "adaptive" log-bisection algorithm (`experiments/calibrate_lambda_adaptive.py`), you are not merely changing "how the search over $\lambda$ behaves." You are actively selecting a specific, highly-tuned sparsity regime. If $L_0$ target validation is run, the resulting $\lambda$ directly dictates the threshold-suppression behavior of the encoder. If you piloted various target $L_0$ values or bisection convergence tolerances on a scratch machine before "pre-registering" target $L_0=32$, you have committed **calibration-hacking**—indirectly optimizing the absorption rate through its primary covariate (sparsity).
*   **Specific Evidence / Experiment to settle it:**
    You must prove that the adaptive calibration path is unique and robust. Report a sensitivity analysis showing how the final absorption rate changes when the target $L_0$ varies from 24 to 40 (your gate's acceptable band). If a small drift in target $L_0$ significantly alters the P1 or P2 results, your calibration is a highly sensitive confounding parameter, and the "outcome-blind" defense is refuted.
*   **Severity:** **MAJOR**
*   **Classification:** `[POST-HOC-ONLY]` (Requires analysis of how the choice of $L_0$ target covaries with the family absorption rate across widths).

---

### PART III: DROPPING CELLS AS A STATISTICAL DEGREE OF FREEDOM

#### Finding 3: Dropping $m=8192$ is a "Budget" Excuse for a Hidden Degree of Freedom
*   **The Claim:**
    > "Amendment 2 — drop m=8192; widths become {2048, 4096, 16384} ... Dropping m=8192 gives ≈12h (~$8), on budget. The sweep still spans 8× in width and retains three points for the monotonicity read." (`notes/prereg-round13b-capacity.md`)
*   **Why it fails / is unsupported:**
    Dropping a cell after locking a pre-registration—specifically a cell that sits *in the middle* of a monotonicity sweep—is a massive statistical degree of freedom. 
    1. **The $4 Excuse:** Claiming that a $4 budget difference on a single L4 GPU is the cause for dropping an entire experimental dimension in a thesis-defining, confirmatory capacity run is highly suspicious. In any professional research context, $4 is statistically indistinguishable from zero cost.
    2. **Monotonicity Smoothing:** Fitting a monotonic curve through 3 points ($2048 \to 4096 \to 16384$) is trivial. Fitting it through 4 points ($2048 \to 4096 \to 8192 \to 16384$) is significantly harder because it exposes the pipeline to mid-regime fluctuations, transition noise, or non-monotonicity. If $m=8192$ had been run and showed a spike or a dip, your P1 monotonicity claim would be falsified. By deleting the interior point under a "budget" pretense, you have smoothed your experimental surface, reducing your ability to reject the null hypothesis of non-monotonicity.
*   **Specific Evidence / Experiment to settle it:**
    You must run the $m=8192$ cell under the identical calibrated conditions and include it in your final report. If the $m=8192$ results violate the monotonicity of the other three points, then Amendment 2 was a confirmatory-saving filter that hid a failure state, and your P1 claim must be downgraded.
*   **Severity:** **MAJOR**
*   **Classification:** `[PREREG-VIOLATION]` (Altering the cells of a locked design based on post-lock runtime or pilot observations constitutes a protocol violation; the $m=8192$ cell must be run and reported).

---

### PART IV: AUDITABLE CRYPTOGRAPHIC PROTOCOL SPECIFICATION

To elevate your pre-registration process from a "theatrical performance" to a mathematically verifiable lock, you must publish a public **Claim Ledger** backed by external cryptographic artifacts.

#### The Four Mandated Artifacts for External Auditing

To make future rounds resilient to hostile scrutiny, the repository must automatically generate and export the following:

```
+---------------------------------------------------------------------------------------+
|                                EXTERNAL AUDIT PROTOCOL                                |
+---------------------------------------------------------------------------------------+
|  1. IMMUTABLE THIRD-PARTY TIMESTAMP (The Genesis Lock)                                 |
|     - Generate a SHA-256 of the pre-registration note and frozen evaluator.           |
|     - Commit and push this hash to a public, timestamped, third-party immutable       |
|       ledger (e.g., Open Science Framework (OSF) API with an active Registration ID,   |
|       or a transaction on a public blockchain).                                       |
|                                                                                       |
|  2. RAW WEIGHTS CONTENT-ADDRESSABLE PROVENANCE (SHA-256 Lock)                         |
|     - For every trained SAE weight file (.pt), record its SHA-256 hash in a signed    |
|       manifest.                                                                       |
|     - The weights must be stored in a WORM (Write-Once-Read-Many) cloud bucket        |
|       (e.g., GCS with Object Retention/Bucket Lock enabled).                          |
|     - The creation timestamps of these objects in the GCS metadata must be            |
|       strictly posterior to the OSF immutable registration timestamp.                 |
|                                                                                       |
|  3. FULL EXECUTION AND CALIBRATION TRACE LOGS (No Deletions)                           |
|     - The "Anti-contamination delete" step must be outlawed. All logs, including      |
|       every single bisection step of the adaptive calibration run (every lambda and    |
|       its resulting intermediate L0), must be written to an un-editable,               |
|       content-addressed JSON file.                                                    |
|     - This proves that calibration was indeed outcome-blind and did not involve       |
|       hidden "warm starts" or manual parameter resets.                                |
|                                                                                       |
|  4. REPRODUCIBILITY SEED MANIFEST & INTEGRITY SIGNATURE                               |
|     - Provide an execution container or a locked Docker manifest containing the       |
|       exact CUDA, PyTorch, and hardware specifications (e.g., NVIDIA driver L4 build). |
|     - Include a verifiable execution script that can run the calibration and training  |
|       on a clean cloud VM from scratch, automatically verifying that the intermediate |
|       lambdas match the reported lambdas to 5 decimal places.                        |
+---------------------------------------------------------------------------------------+
```

Without these four artifacts, candidate, your open-science pre-registration is merely a set of files in a Git repository that you control—and your defense of its integrity rests entirely on your personal credibility. As your external examiner, I do not have credit in that currency. 

How do you defend the validity of your amendments when you have left yourself the perfect, un-auditable backdoor to shape your results?
