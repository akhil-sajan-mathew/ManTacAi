# 3.5 ALGORITHMS

### 3.5.1 Algorithm 1: Conversational Data Preprocessing
The preprocessing pipeline standardizes raw text sequences to ensure uniform input distributions for the transformer model.
1. **Input:** Raw conversational string stream $S_{raw}$.
2. **Sanitization:** Apply regex-based filtering to remove non-semantic noise (e.g., arbitrary special characters) while preserving emotionally charged punctuation. Convert $S_{raw}$ to lower case representation $S_{clean}$.
3. **Subword Tokenization:** Utilize a Byte-Pair Encoding (BPE) tokenizer to segment $S_{clean}$ into a sequence of subword tokens $T = \{t_1, t_2, \dots, t_N\}$.
4. **Sequence Formatting:** Truncate or pad $T$ to a fixed maximum sequence length (e.g., $N=512$).
5. **Output:** A standardized integer vector of `input_ids` and a corresponding binary `attention_mask` ($A_{m}$), which restricts the model from applying self-attention to padded tokens.

### 3.5.2 Algorithm 2: Contextual Encoding
This stage transforms discrete integer tokens into continuous, dense mathematical representations capturing high-level linguistic semantics.
1. **Input:** `input_ids` and `attention_mask` ($A_{m}$).
2. **Transformer Integration:** Feed the inputs into the fine-tuned base transformer architecture (e.g., DistilRoBERTa).
3. **Self-Attention Mechanism:** The model computes multi-head attention arrays, weighing the relevance of each token $t_i$ against all other $t_j$ in the sequence.
4. **Vector Embedding:** Extract the hidden state associated with the `[CLS]` token from the final transformer layer, producing a continuous representation.
5. **Output:** A dense 768-dimensional context-aware encoded vector $\mathbf{h} \in \mathbb{R}^{768}$, which encapsulates the holistic contextual meaning of the input message.

### 3.5.3 Algorithm 3: Manipulation Detection (Classification)
The classification algorithm maps the extracted contextual embeddings into an 18-class manipulation taxonomy.
1. **Input:** The 768-dimensional encoded vector $\mathbf{h}$.
2. **Feature Extraction:** Pass $\mathbf{h}$ through a fully connected dense layer combined with a dropout normalization algorithm to prevent overfitting.
3. **Softmax Output Layer:** Project the normalized features onto the 18 specific class dimensions and apply the softmax activation function to compute a probability distribution:
   $$ P(y_i | \mathbf{h}) = \frac{e^{\mathbf{W}_i \mathbf{h} + b_i}}{\sum_{j=1}^{18} e^{\mathbf{W}_j \mathbf{h} + b_j}} $$
4. **Label Inference:** Determine the peak probability score $P_{\text{max}} = \max(P)$ and derive the predicted manipulation label $\hat{y} = \arg\max(P)$.
5. **Output:** Predicted manipulation class $\hat{y}$ paired with its confidence score $P_{\text{max}}$.

### 3.5.4 Algorithm 4: Context Analysis & Cycle of Abuse Detection
A stateful tracking engine that monitors transitions between messages over a rolling temporal window, mapping them to the clinical phases of abuse.
1. **Input:** Time-sequenced stream of predicted classes $\hat{y}$ and corresponding context scores.
2. **State Machine Initialization:** Initialize the sequence engine at the *Normal* phase baseline.
3. **Temporal Mapping:** Apply sequential transition logic to evaluate deviations from the baseline towards *Tension Building*, *Explosion*, and *Honeymoon* phases based on predefined label severities.
4. **Deterministic Override (Circuit Breaker):** To prevent temporal manipulation or rapid false apologies from artificially resetting the cycle, the system employs an absolute lock mechanism. Specifically, it securely locks the system into an "Explosion" state if the Risk Score ($R$) breaches critical safety limits:
   $$ \text{State} = \text{"Explosion"} \quad \text{if} \quad R > 0.85 $$
5. **Output:** The currently active manipulation phase (Cycle State) derived from the persistent behavioral sequence.

### 3.5.5 Algorithm 5: Risk Scoring Mechanism
A dynamic threat quantification system that calculates localized conversation risk by fusing raw probabilities with contextual dampeners and clinical severity arrays.
1. **Input:** Maximum predicted class probability ($P_{\text{max}}$) and the predicted class label $\hat{y}$.
2. **Severity Weight Assignment:** Assign a predefined clinical Severity Weight ($W_s$) corresponding to $\hat{y}$ mapping the psychological damage multiplier.
3. **Real-Time Risk Equation ($R$):** Continuously quantify threat levels using $P_{\text{max}}$, $W_s$, and a contextual Dampening Factor ($D_f$) applied to reactive speakers:
   $$ R = P_{\text{max}} \times W_s \times D_f $$
4. **Anti-Dampening (Trojan Horse) Protocol:** Minimizes false positives in benign contexts by mathematically dampening risk scores ($D_f = 0.1$). If semantic threat thresholds ($>0.90$) are breached, dampening is neutralized ($D_f = 1.0$), unlocking the full Risk Score.
5. **Risk Stratification:** Classify $R$ against standard operational thresholds (Low, Medium, Critical Risk).
6. **Output:** The final, heavily calibrated risk quotient $R$ and its corresponding threat level tier.

### 3.5.6 Algorithm 6: Semantic Threat Engine & Guardrails
A secondary zero-shot protection layer running in parallel, designed to catch highly evasive or structurally "coded" threats that circumvent the standard softmax classifier.
1. **Input:** Extracted 768-dimensional vector embedding ($\mathbf{v}$) from the transformer's hidden states.
2. **Centroid Anchor Comparison:** Initialize a predefined matrix of known critical threat centroids ($\mathbf{c}$) mapped in the semantic space representing severe coercion vectors.
3. **Similarity Assessment:** Utilizing Cosine Similarity against known threat centroids ($\mathbf{c}$) to identify evasive or coded language:
   $$ \text{Similarity} = \frac{\mathbf{v} \cdot \mathbf{c}}{ ||\mathbf{v}|| \cdot ||\mathbf{c}|| } $$
4. **Rule-Based Trigger:** If the computed similarity exceeds an established bounds threshold, trigger an absolute mathematical override bypassing standard classification logic.
5. **Output:** Safety alert execution and overridden clinical classification label bridging gaps in standard inference.

### 3.5.7 Algorithm 7: Real-Time Analysis Pipeline
The operational orchestrator governing the sequential execution of models over a live web-socket or API data stream.
1. **Input:** A continuous JSON payload containing streaming chat chronologies.
2. **Concurrency Processing:** Instigate asynchronous task routing to apply Algorithms 1 through 6 against inbound message bursts with zero blocking.
3. **State Synthesis:** Route individual message variables to global Context Controllers which dynamically aggregate and update interaction states over the rolling window.
4. **Immediate Distribution:** Cascade updated JSON response matrices containing $R$ values, Cycle States, and Guardrail logic directly back to the front-end rendering engines.
5. **Output:** End-to-end labeled conversation schema rendered instantaneously on the client user interface $O(\text{1})$ complexity latency.

### 3.5.8 Algorithm 8: Visualization and Report Generation
The data summation framework rendering complex longitudinal manipulation analytics into comprehensible forensic formats.
1. **Input:** Processed operational array of sequential analysis matrices spanning an entire conversation window.
2. **Aggregation Arithmetic:** Compute macro-level tactic groupings, generating tactic frequency distributions, timeline volatility coordinates, and DARVO (Deny, Attack, Reverse Victim and Offender) indexes.
3. **Canvas Translation:** Pass distribution coefficients to standard library routines to draw predictive probability arcs, severity weighting charts, and historical heatmaps.
4. **Documentation Compilation:** Aggregate visual nodes and classified strings into structured PDF/Word forensic templates via server-side generation buffers.
5. **Output:** A comprehensive, statically compiled Forensic Analysis Report.
