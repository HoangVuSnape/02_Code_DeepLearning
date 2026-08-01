# Prompt TẠO ẢNH cho Gemini (v2 - Đã sửa lỗi) — 12 kiến trúc trong `03_kien_truc_model.html`

## Quy tắc toàn cục (BẮT BUỘC trong mọi Prompt v2)

1. **Tỉ lệ khung hình (Aspect Ratio):** Luôn yêu cầu `16:9` landscape (hoặc `4:3`), mở rộng chiều ngang canvas, không để lãng phí khoảng trắng trên/dưới.
2. **Ký hiệu toán chuẩn Unicode:** Sử dụng trực tiếp `→`, `⊙`, `⊕`, `⊗`, `σ`, `π`, `α`, `Σ`, `∈`, `·`, `×`. KHÔNG dùng ASCII thay thế (`->`, `x`, `*`, `sigma`, `pi`, `alpha`, `Sum`, `in`, `.`).
3. **Cấm in chữ chỉ dẫn meta:** Không bao giờ render các từ như `UPPER`, `LOWER`, `BRANCH`, `HALF`, `ZONE`, `STRUCK THROUGH`, `HOT`, `ACTIVE`, `STYLE`, `CONSTRAINTS` ra màn hình.
4. **Ràng buộc định lượng:** Độ dài thanh vector, số ô vuông, chiều cao cột bar chart phải đúng tỉ lệ với con số in trên nhãn (`128:64:192`, `0.42` cao gấp 3.5 lần `0.12`).
5. **Đồng nhất màu nền & X-quang:** Tất cả ảnh dùng chung màu nền `#F6F8FB` và 1 mẫu ảnh X-quang ngực duy nhất.

---

## PROMPT 0 — Toàn cảnh pipeline (v2)

```
A clean, modern technical infographic diagram of a multimodal medical Visual Question Answering neural network pipeline, flat vector illustration style, horizontal left-to-right data flow, 16:9 aspect ratio filling the full canvas width.

TOP BRANCH (teal #0E7490): chest X-ray thumbnail labeled "Image [B,3,224,224]" → rounded box "CNN / ResNet-18" (+SE-block optional) → tag "v_img [B,128]" → dashed box "Projector (MLP)" → tag "z_img [B,P]".

BOTTOM BRANCH (amber #D97706): speech-bubble labeled "Question tokens [B,32]" → rounded box "BiLSTM / Transformer" (+temporal attention optional) → tag "v_txt [B,64]" → dashed box "Projector (MLP)" → tag "z_txt [B,P]".

CENTER MERGE (purple #7C3AED): both projector outputs merge into box "Concat" labeled "[B,2P]". Below it, a thin dashed bypass line shows direct concatenation "v_img + v_txt → [B,192]" for no-projector mode.

RIGHT (blue #2563EB then green #15803D): "GRU Decoder + Cross-Attention" box with subtitle "logits [B,T,V_answer]" → "Greedy / Beam Search" → green rounded tag showing generated answer: "right lung".

STYLE: flat vector infographic, off-white background #F6F8FB, rounded white boxes with 2px colored outlines, soft subtle drop shadows, thin blue arrows with sharp heads, geometric sans-serif type. Top-left legend: teal = image, amber = text, purple = fusion, blue = decoder, green = answer.

CONSTRAINTS: 16:9 widescreen layout without large empty top/bottom margins. All labels in clean short English Unicode typography. No meta prompt instructions as text labels. No photorealism, no 3D perspective.
```

---

## PROMPT ① — CustomCNN (v2)

```
A clean technical infographic illustrating a 3-layer custom convolutional neural network used as a medical image encoder, flat vector style, horizontal left-to-right flow, 16:9 landscape aspect ratio, teal #0E7490 as dominant color.

MAIN VISUAL IDEA: feature maps drawn as stacks of layered slabs shrinking in height by 50% at each step (224 → 112 → 56 → 28) while stack thickness increases (4 slabs → 6 slabs → 8 slabs).

SEQUENCE OF BOXES CONNECTED BY THIN TEAL ARROWS (→):
1. small chest X-ray thumbnail, label "Input 3×224×224"
2. block "Conv 3→32 + BN + ReLU" with sublabel "MaxPool → 32×112×112", beside it a stack of 4 thin slabs labeled "32 ch"
3. block "Conv 32→64 + BN + ReLU" with sublabel "MaxPool → 64×56×56", beside it a stack of 6 medium slabs labeled "64 ch"
4. block "Conv 64→128 + BN + ReLU" with sublabel "MaxPool → 128×28×28", beside it a stack of 8 thick slabs labeled "128 ch"
5. dashed box "SE-Block (optional)"
6. box "AvgPool + LayerNorm"
7. final rounded teal tag "v_img [B,128]" drawn as a horizontal strip of small squares representing a 128-dim vector.

STYLE: flat vector infographic, off-white background #F6F8FB, white boxes with 2px teal outlines, geometric sans-serif type, thin arrows with sharp arrowheads.

CONSTRAINTS: 16:9 landscape canvas, full width. Text labels in exact Unicode typography. No duplicate boxes. No photorealism.
```

---

## PROMPT ② — ResNet-18 đóng băng (v2)

```
A technical infographic contrasting a FROZEN pretrained backbone against a small TRAINABLE head, flat vector style, horizontal left-to-right flow, 16:9 landscape aspect ratio.

LEFT: chest X-ray thumbnail, label "Image [B,3,224,224]", note "ImageNet norm".

CENTER (FROZEN BACKBONE): large icy-blue dashed container labeled "ResNet-18 backbone (pretrained)" with output tag "[B,512,7,7]", marked with a small padlock icon and caption "FROZEN — no gradient update". Inside, draw 4 sequential stage boxes "layer1 → layer2 → layer3 → layer4" connected in series with arrows. Above them, a residual skip curved arrow bypassing two conv blocks to a ⊕ symbol. A red backward gradient arrow is stopped at the outer edge of the frozen container with a red 🚫 stop icon labeled "no gradient".

THEN: dashed box "SE-Block (optional)" → box "AdaptiveAvgPool → [B,512]".

RIGHT (TRAINABLE HEAD): warm green #15803D active box labeled "Linear(512→128) + LayerNorm" with subtitle "TRAINABLE — only this part learns", emitting final teal tag "v_img [B,128]". All text labels rendered horizontally.

STYLE: flat vector infographic, off-white background #F6F8FB, teal #0E7490 for image branch, icy grey-blue for frozen region, green #15803D for trainable region, blue #2563EB arrows, geometric sans-serif typography.

CONSTRAINTS: 16:9 landscape. Do NOT render prompt instruction words like "STRUCK THROUGH", "HOT", or "ACTIVE" as text in the image. Proper Unicode arrows (→).
```

---

## PROMPT ③ — SE-Block (v2)

```
A technical infographic explaining the Squeeze-and-Excitation channel attention block, flat vector style, 5-step horizontal left-to-right flow, 16:9 landscape.

FIVE STEPS CONNECTED BY THIN BLUE ARROWS (→):
1. "Feature map F [B,C,H,W]" — stack of EXACTLY 6 isometric channel slices, ALL SAME uniform pale teal color.
2. "Squeeze: Global Avg Pool → [B,C]" — collapsing stack into a single column.
3. "Excite: FC(C→C/4) → ReLU → FC(C/4→C) → Sigmoid" — below it, a bar chart of EXACTLY 6 vertical bars of heights corresponding to values: 0.9, 0.2, 0.7, 0.95, 0.1, 0.5, with caption "s ∈ [0,1]^C".
4. "Scale: s ⊙ F" — a central circle containing the ⊙ element-wise Hadamard multiply symbol (not ×).
5. "F' reweighted [B,C,H,W]" — stack of EXACTLY 6 channel slices where color saturation matches weight: 0.95 slice is vivid saturated teal, 0.1 slice is ghost-grey.

KEY LINK: 6 thin dashed guide lines connect the top of each of the 6 bars in step 3 directly to its corresponding slice in step 5.

Bottom caption: "Useful channels amplified, noisy channels suppressed".

STYLE: flat vector infographic, off-white background #F6F8FB, teal #0E7490 accent, blue #2563EB arrows, green #15803D final output, white boxes with thin outlines.

CONSTRAINTS: 16:9 landscape. Exactly 6 slices in F, 6 bars in step 3, and 6 slices in F'. Element-wise multiply symbol MUST be ⊙. Unicode typography.
```

---

## PROMPT ④ — BiLSTM (v2)

```
A technical infographic of a Bidirectional LSTM text encoder unrolled over 5 time steps, flat vector style, 16:9 landscape aspect ratio, amber #D97706 dominant accent.

LEFT: box "Question tokens [B,32]" → box "Embedding(V,64)" → tag "[B,32,64]".

CENTER (UNROLLED BILSTM):
- Middle row: 5 word chips "is", "there", "pleural", "effusion", "?"
- Top row: 5 LSTM cell boxes connected LEFT-TO-RIGHT with arrows pointing right (→), labeled "forward h→" (deep amber).
- Bottom row: 5 LSTM cell boxes connected RIGHT-TO-LEFT with arrows pointing left (←), labeled "backward h←" (medium amber).
- Left-side note: "hidden 32 per direction → 64 total".

RIGHT (POOLING BRANCHES):
- Solid box "concat(last h→, first h←) → [B,64]" with tag "no-attention mode". An arrow exits from the FIRST backward LSTM cell (leftmost) AND the LAST forward LSTM cell (rightmost), both connecting into this concat box.
- Dashed box "Temporal Attention over [B,32,64] → [B,64]" with tag "attention mode", receiving a bus line collecting states from all 5 timesteps.
Both converge into "LayerNorm" → final amber tag "v_txt [B,64]". All text labels printed horizontally.

STYLE: flat vector infographic, off-white background #F6F8FB, white rounded boxes with 2px amber outlines, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape layout filling width. Backward chain MUST connect to concat box. Correct Unicode arrows (→, ←).
```

---

## PROMPT ⑤ — Transformer Encoder (v2)

```
A technical infographic of a Transformer encoder text encoder, flat vector style, 16:9 landscape, amber #D97706 dominant, horizontal main flow on top and detail panel below.

TOP ROW (MAIN FLOW):
"Question tokens [B,32]" → "Embedding + Positional Encoding → [B,32,64]" → stacked layer box "2× TransformerEncoderLayer (d=64, heads=4, ff=128)" → output "[B,32,64]" → fork into "masked-mean (no-attn)" and "attention pooling" → "LayerNorm" → final amber tag "v_txt [B,64]".

BOTTOM LEFT (ZOOM CALLOUT):
Inside dashed callout frame: "Multi-Head Self-Attention (4 heads)" → "LayerNorm" → "Feed-Forward 64→128→64" → "LayerNorm". Curved residual skip lines bypass sublayers, rejoining at ⊕ addition circle symbols before LayerNorm.

BOTTOM RIGHT (HEATMAP):
A 5×5 attention heatmap grid labeled "is", "there", "pleural", "effusion", "?". The cell at row "effusion", column "pleural" is the DARKEST saturated amber cell (showing cross-token medical attention), while diagonal cells are medium shaded. Caption: "every token attends to every token, in parallel".

STYLE: flat vector infographic, off-white background #F6F8FB, amber outlines, blue #2563EB arrows, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape canvas. Heatmap darkest cell MUST be (effusion, pleural). Unicode typography. No duplicate x2 labels.
```

---

## PROMPT ⑥ — Temporal Attention (v2)

```
A technical infographic explaining additive temporal attention pooling over a sequence, flat vector style, 16:9 landscape, amber #D97706 dominant.

TOP FLOW:
"outputs [B,T,H]" → "Linear(H→1) → scores [B,T]" → "mask PAD + softmax (αt [B,T], Σα = 1)" → "Σ αt · ht (weighted sum)" → final green tag "context [B,H]".

BOTTOM (BAR CHART VISUALIZATION):
A row of 7 token chips: "is", "there", "pleural", "effusion", "?", "<pad>", "<pad>".
Above each token stands a vertical bar on a SHARED BASELINE with height proportional to attention weight:
- "effusion": tallest bar, value 0.42 (deep amber)
- "pleural": second tallest bar, value 0.31
- "?": bar value 0.12
- "is": bar value 0.08
- "there": bar value 0.07
- two "<pad>" tokens: EXACTLY ZERO HEIGHT (completely flat on baseline), greyed out with red diagonal strike-through and red caption "PAD → α = 0 (masked)".
Thin arrows from the top of the 5 non-pad bars converge into a single rounded green box on the right labeled "context vector [B,H]".

Bottom caption: "Important clinical keywords receive high weight; PAD tokens receive exactly zero weight."

STYLE: flat vector infographic, off-white background #F6F8FB, white boxes with amber outlines, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape. PAD bars MUST have zero height (flush with baseline). Exact weights 0.42, 0.31, 0.12, 0.08, 0.07 (sum = 1.00). Unicode symbols (α, Σ, ·, →).
```

---

## PROMPT ⑦ — Projector / MLP Mapper (v2)

```
A technical infographic explaining dual projector MLPs mapping image and text modalities into a shared latent space, flat vector style, 16:9 landscape, purple #7C3AED dominant.

TOP HALF (FLOW DIAGRAM):
On the left, two stacked input tags: teal "v_img [B,128]" and amber "v_txt [B,64]".
Each connects to its OWN SEPARATE purple box stacked vertically:
- Upper purple box: "Img Projector: Linear(128→P) → GELU → Linear(P→P) → LayerNorm" → output purple tag "z_img [B,P]"
- Lower purple box: "Txt Projector: Linear(64→P) → GELU → Linear(P→P) → LayerNorm" → output purple tag "z_txt [B,P]"
These are two INDEPENDENT modules with NO shared weights. Both purple tags merge into a purple box "Concat [B,2P]".

BOTTOM HALF (LATENT ALIGNMENT COMPARISON):
Two square scatter-plot panels side by side with a purple arrow labeled "Projector MLPs":
- LEFT PANEL ("Before projector — unaligned spaces"): teal dots clustered in top-left, amber dots clustered in bottom-right, separated by a wide empty gap. Caption: "naive concat — unaligned modalities".
- RIGHT PANEL ("After projector — shared latent space"): teal and amber dots interleaved and evenly mixed with thin dashed lines connecting matching pairs. Caption: "aligned — shared multimodal latent space (conceptual illustration)".

STYLE: flat vector infographic, off-white background #F6F8FB, teal #0E7490, amber #D97706, purple #7C3AED, white panels, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape aspect ratio. Two SEPARATE projector boxes, never a single box. Concat box MUST be purple #7C3AED. Do NOT print "UPPER HALF" or "LOWER HALF" as text labels. Unicode typography.
```

---

## PROMPT ⑧ — Fusion bằng Concatenation (v2)

```
A technical infographic explaining feature concatenation (early fusion) of two vector modalities, flat vector style, 16:9 landscape.

CORE VISUAL METAPHOR (VECTOR STRIPS):
- Top strip: a long teal strip of 24 small squares, labeled "v_img [B,128]", bracket underneath reading "128 dims".
- Middle strip: an amber strip of 12 small squares (EXACTLY HALF LENGTH of top strip), labeled "v_txt [B,64]", bracket reading "64 dims".
- Between them: a purple rounded rectangle box labeled "concat".
- Result strip: ONE single long strip of 36 small squares (total length = sum of top two strips). Left 24 squares are teal, right 12 squares are amber (EXACTLY 2:1 length ratio). Labeled "fused [B,192]", bracket reading "192 dims".
- Right: a blue arrow line exiting into a box labeled "GRU Decoder".

Bottom caption: "Simple and interpretable concatenation (128 + 64 = 192 dims). No cross-modal interaction at feature level."

STYLE: flat vector infographic, off-white background #F6F8FB, teal #0E7490, amber #D97706, purple #7C3AED, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape. Strip length ratio MUST be 24 squares (teal) + 12 squares (amber) = 36 squares (fused). Use word "concat" in a box (no large + plus symbol). Blue arrow must be a drawn line, not text. Unicode typography.
```

---

## PROMPT ⑨ — Gated Attention (v2)

```
A technical infographic explaining a learned sigmoid gate applied element-wise to a fused feature vector, flat vector style, 16:9 landscape, purple #7C3AED dominant.

TOP BANNER FORMULA:  z' = z ⊙ σ(W z)

FLOW DIAGRAM:
1. Left: "fused z [B,D]" drawn as a horizontal strip of 12 uniform purple squares.
2. Path splits into two:
   - Upper path: box "Linear(D→D) → σ" → "gate [B,D], values ∈ [0,1]" drawn as a second strip of 12 squares with distinct intensities. Sample numeric values printed above squares: 0.92 above square #1 (vivid purple), 0.11 above square #2 (faint), 0.78 above square #3, 0.05 above square #4 (ghost-grey).
   - Lower path: z travels straight through as a main vector line.
3. Both paths meet at a central circle containing the ⊙ element-wise multiply symbol.
4. Right: "z' [B,D]" — output strip of 12 squares where each square's color intensity matches its gate weight 1-to-1: square #1 is vivid purple, square #2 and #4 are faint ghost-grey.

ANNOTATIONS:
- Green arrow pointing to vivid square #1: "useful dim kept"
- Grey arrow pointing to ghost square #4: "noisy dim suppressed"

Bottom note: "Same dimension in and out (D→D) — element-wise gating regulates feature flow".

STYLE: flat vector infographic, off-white background #F6F8FB, purple #7C3AED, blue #2563EB arrows, green #15803D accents, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape. Formula MUST use ⊙ and σ (z' = z ⊙ σ(W z)). Circle operation MUST be ⊙. Gate values and output strip intensities MUST match 1-to-1. Unicode typography.
```

---

## PROMPT ⑩ — GRU Decoder generative + Cross-Attention (v2)

```
A detailed technical infographic of an autoregressive GRU decoder with cross-attention generating a medical VQA answer token by token, flat vector style, 16:9 landscape, blue #2563EB dominant.

TOP BANNER FORMULA:  p(y₁:T | I, q) = ∏_{t=1}^T p(y_t | y_{<t}, I, q)

LEFT INITIALIZATION:
Tag "fused [B,192] or [B,2P]" → box "Linear(fused→128)" → "hidden h0 [B,128]" feeding into GRU cell #1.

CENTER UNROLLED TIMELINE (4 GRU CELL BOXES IN A ROW):
Connected left-to-right by hidden state arrows "[B,128]".
Under each cell, generated token appears in a rounded green chip:
- Step 1: input "<bos>" → outputs token chip "right"
- Step 2: input "right" → outputs token chip "lung"
- Step 3: input "lung" → outputs token chip "<eos>"
- Step 4: ghosted cell with caption "… until <eos> or MAX_LEN"
CRITICAL AUTOREGRESSIVE FEEDBACK ARROWS: Curved arrows loop FORWARD from each generated token chip into the next GRU cell's input (e.g., token chip "right" at step 1 loops into GRU cell #2 input).
Each GRU cell receives: "prev token emb [B,64]" and "cross-attn context [B,128]", emitting upward "Linear(128→V_answer)" → "logits [B,V_answer]".

TOP CROSS-ATTENTION PANELS:
- Teal panel: "image spatial tokens [B,49,128]" rendered as 7×7 grid of patches on faint chest X-ray.
- Amber panel: "question hidden states [B,32,64]" rendered as word chips "is", "there", "pleural", "effusion", "?".
At step 2 (generating "lung"), thin query arrows go UP from GRU cell #2 into both panels, context arrows come DOWN. 2–3 lung patches GLOW BRIGHT TEAL, and chips "pleural" and "effusion" GLOW BRIGHT AMBER.

BOTTOM SPLIT NOTE:
- LEFT: "TRAIN — teacher forcing: input is ground-truth token, loss = cross-entropy"
- RIGHT: "INFERENCE — greedy / beam search: input is model's own previous token"

STYLE: flat vector infographic, off-white background #F6F8FB, teal #0E7490, amber #D97706, blue #2563EB, green #15803D chips, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape. Generated tokens MUST spell "right" -> "lung" -> "<eos>". Autoregressive feedback arrows MUST point FORWARD to next step. Spatial tokens tag MUST be [B,49,128] (7×7 grid). Unicode typography.
```

---

## PROMPT ⑪ — Reinforcement Learning: REINFORCE Self-Critical (v2)

```
A technical infographic of the REINFORCE Self-Critical Sequence Training loop, flat vector style, 16:9 landscape, crimson #BE123C dominant accent, off-white background #F6F8FB (matching full set).

CLOSED LOOP COMPOSITION:
1. Far left: box "Model after SFT (policy π)" emitting "logits".
2. Splits into two parallel branches, one drawn above the other:
   - the sampled path (crimson #BE123C border): "Sample answer ~ Categorical(logits)" → arrow ↓ → "reward(sample) = 0.5·EM + 0.5·F1"
   - the baseline path (muted grey border): "Greedy answer = argmax" → arrow ↓ → "reward(greedy)" (caption: "SELF-CRITICAL BASELINE — no value network")
3. Both branches converge to: "Advantage = r(sample) − r(greedy)"
4. Single-line Loss box: "Loss = − ( log π(sample) · advantage ).mean()" (with single minus sign).
5. A THICK DASHED CURVED RETURN ARROW sweeps from Loss box all the way back around to Model box, labeled "policy update (lr 1e-5, ~5 epochs)".

BELOW LOOP (EXAMPLE OUTCOMES SIDE BY SIDE):
- CASE A: bar "sample: right lung (r=0.9)" vs bar "greedy: lung (r=0.5)" → "advantage = +0.4" → GREEN UPWARD arrow "push sample prob UP"
- CASE B: bar "sample: no (r=0.1)" vs bar "greedy: right lung (r=0.9)" → "advantage = −0.8" → RED DOWNWARD arrow "push sample prob DOWN"

Bottom notes with orange left border:
"Why RL: Cross-entropy optimizes token likelihood; RL directly optimizes non-differentiable EM/F1 metrics."
"Order matters: Supervised Fine-Tuning (SFT) first, then SCST RL fine-tuning."

STYLE: flat vector infographic, off-white background #F6F8FB, crimson #BE123C sampled branch, grey baseline branch, green #15803D and red accents, geometric sans-serif type.

CONSTRAINTS: 16:9 landscape. Background MUST be off-white #F6F8FB (not cream, not beige). Loss MUST have a SINGLE minus sign: "Loss = −( log π(sample) · advantage ).mean()". Use the middle dot · in 0.5·EM and log π(sample) · advantage — never a full stop. In the greedy box, put "argmax" and "reward(greedy)" side by side or with a ↓ arrow — NEVER stack them with a horizontal rule between (it reads as a fraction bar). All reward bars share ONE baseline and one implied y-axis: a bar labeled r=0.9 must be exactly 9× the height of a bar labeled r=0.1, and the two r=0.9 bars in CASE A and CASE B must be identical in height. Lay CASE A and CASE B out in the SAME reading order (bars → advantage → arrow). Unicode typography.
```

---

## PROMPT BONUS A — Poster tổng hợp 1 trang (v2)

```
A large one-page scientific poster infographic summarizing an entire multimodal medical VQA architecture, flat vector style, landscape 16:9 aspect ratio, off-white background #F6F8FB, designed for a thesis defense slide.

The ONLY text allowed in the title banner is exactly: "Multimodal Medical VQA Architecture Overview".
Do not add, translate, paraphrase or invent any other word in the title.

FOUR PANELS CONNECTED BY ONE CONTINUOUS MAIN FLOW ARROW (LEFT TO RIGHT).
Each panel carries a heading; the heading text is given below in quotes and must be
reproduced EXACTLY, with no numbering prefix of any kind:

Panel 1, heading "ENCODERS" (left):
- Top branch (teal #0E7490): chest X-ray → "CustomCNN" OR "ResNet-18 frozen" → dashed "SE-Block" → "v_img [B,128]"
- Bottom branch (amber #D97706): question chips reading "is there pleural effusion ?" → "BiLSTM" OR "Transformer Encoder" → dashed "Temporal Attention" → "v_txt [B,64]"

Panel 2, heading "ALIGNMENT & FUSION" (center-top, purple #7C3AED):
two separate boxes "Img Projector 128→P" and "Txt Projector 64→P" → "Concat [B,2P] or [B,192]" → dashed "Gated Attention"

Panel 3, heading "GENERATION" (center-right, blue #2563EB):
"GRU Decoder (hidden 128) + Cross-Attention" → "logits [B,T,V_answer]" → "Greedy / Beam" → green chip "right lung"

Panel 4, heading "RL FINE-TUNE" (far right, crimson #BE123C):
compact closed loop "Sample vs Greedy Baseline → Advantage → Policy Gradient", tagged "runs after SFT"

MAIN FLOW ARROW: one single continuous thick arrow. Every arrowhead must LAND ON the
border of the panel it points at — no arrowhead may end in empty space, and no arrow
segment may stop short of its target or overlap another segment's tail.

LEGEND (top-right), five rows, each swatch a VISIBLY DIFFERENT hue:
teal #0E7490 = Image Encoder · amber #D97706 = Text Encoder · purple #7C3AED = Fusion ·
blue #2563EB = Decoder · crimson #BE123C = RL Loop.

BADGE NUMBERS — assign exactly one badge per module, no duplicates, no omissions:
① CustomCNN · ② ResNet-18 frozen · ③ SE-Block · ④ BiLSTM · ⑤ Transformer Encoder ·
⑥ Temporal Attention · ⑦ the two Projectors · ⑧ Concat · ⑨ Gated Attention ·
⑩ GRU Decoder + Cross-Attention · ⑪ RL loop.
Do NOT place a badge on a tensor tag such as "v_img [B,128]".

Footer line: "Multimodal VQA-RAD Architecture · Master Thesis TDTU".

STYLE: flat vector poster infographic, white cards with thin colored borders and soft shadows, strong visual hierarchy, geometric sans-serif typography.

CONSTRAINTS: 16:9 landscape format filling full canvas width. The title MUST read exactly "Multimodal Medical VQA Architecture Overview" — no garbled or invented words. All four panel headings must be present; none may be omitted. Legend swatches for Text Encoder and RL Loop must be clearly different colours (amber vs crimson), never two shades of the same orange. No label may overlap a panel border. All text in clean Unicode English.
```
