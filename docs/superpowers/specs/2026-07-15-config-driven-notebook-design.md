# Thiết kế: Notebook thực nghiệm config-driven + lưu HF theo sub-folder

Ngày: 2026-07-15

## Mục tiêu
Thay quy trình hiện tại (25+ cell, mỗi cell lặp lại toàn bộ tham số CLI) bằng một
notebook **config-driven**: mọi hằng số (epochs, lr, batch, tên project Comet, repo HF)
nằm ở **một cell CONFIG duy nhất**; các run đọc từ đó. Đổi cấu hình = sửa 1 cell, không
đụng cell run, không cần commit code. Checkpoints/metrics lưu về **1 repo HF** nhưng chia
**mỗi run một sub-folder** cho gọn.

## Không làm (non-goals)
- Không refactor `train.py`/`eval.py` thành thư viện import-được (giữ interface CLI hiện có).
- Không đổi logic model/eval/metrics.
- Không xoá `run_pipeline.ipynb` cũ (giữ làm backup).

## Cách thực thi
Hướng A — **config-driven + subprocess**. Notebook dựng danh sách tham số rồi gọi
`train.py`/`eval.py` qua `subprocess.run([...])` (list args, không qua shell). Giữ nguyên
code đã kiểm thử, chỉ thêm lớp điều phối mỏng trong notebook.

## Phần 1 — File mới `run_experiments.ipynb`

Bố cục cell:
1. **Setup**: clone/pull repo (fetch+reset origin/main), `pip install`, load secrets, bật
   `HF_HUB_ENABLE_HF_TRANSFER`, cài `hf_xet`.
2. **CONFIG** (cell chỉnh tay duy nhất):
   - `EPOCHS, LR, PATIENCE, RL_EPOCHS, RL_LR`
   - `BATCH = {"light":128, "heavy":64, "gpt2":32}`
   - `COMET_PROJECT`, `HF_REPO`
   - `USE_HF_PUSH, USE_COMET, USE_DISCORD` (bool)
3. **RUN MATRIX**: `RUNS` = dict nhóm → list run. Mỗi run là dict chỉ khai phần khác nhau:
   `name, img, txt, dec, batch(key), attn(bool), freeze(None|"image"|"text"|"decoder")`.
   Có thêm `EVAL_ONLY` (R_constrained) và `RL` (D1_scst: load_checkpoint) và `GEMMA` (CSV).
4. **HELPER**: `run_train(cfg)`, `run_eval(cfg)` — dựng args từ CONFIG + cfg, `subprocess.run`.
   `train.py` được gọi với `--epochs {EPOCHS} --lr {LR} ... --batch_size {BATCH[key]}`.
5. **Cell chạy theo nhóm**: `for r in RUNS["A1"]: run_train(r)` rồi `for r in RUNS["A1"]: run_eval(r)`.
6. **Backup gộp** + **report** cuối.

Ma trận run (giữ nguyên như hiện tại):
- **A1** (không attention): cnn × {lstm, transformer} × {all, freeze_image, freeze_text,
  freeze_decoder} = 8 run, batch "light".
- **A2**: y hệt A1 nhưng bật `image_attention + text_attention` = 8 run, batch "light".
- **P**: `P_resnet_lstm`, `P_resnet_trans` (resnet18_frozen, attn, batch "heavy");
  `P_pubmed_mlp` (pubmedclip+pubmedbert, batch "heavy"); `P_pubmed_gpt2` (decoder gpt2, batch "gpt2").
- **R**: `R_constrained` — eval-only trên `P_pubmed_gpt2_best` với `--constrained_closed`.
- **D**: `D1_scst` — `train.py --rl --load_checkpoint P_pubmed_gpt2/best.pt`; eval constrained.
- **G**: `gemma-2b-lora`, `gemma-4b-lora` — eval `--predictions_csv`.
- **Smoke**: `smoke_test_run` (cnn/lstm/mlp), `smoke_pubmed`, `smoke_gpt2` — `--no_hf_push` khi eval.

Lưu ý: cờ eval (freeze/attention) phải khớp cờ train để đếm `params_trainable` đúng.

## Phần 2 — Lưu HF theo sub-folder = tên run

Sửa để mỗi run có folder riêng trên repo. Kết quả:
```
A1_lstm_all/best.pt, A1_lstm_all/checkpoint.pt, A1_lstm_all/history.csv, A1_lstm_all/metrics.json
P_pubmed_gpt2/best.pt, ...
```

Thay đổi code:
- `hf_push.push_file(..., path_in_repo=...)`: đã hỗ trợ; caller truyền `f"{run_name}/{basename}"`.
- `hf_push.push_folder(..., path_in_repo=run_name)`: truyền tham số `path_in_repo` cho
  `upload_folder` để đặt file vào sub-folder.
- `callbacks.py`: `on_epoch_end`/`on_run_end` push vào sub-folder `= run_name`.
- `eval.py`: thêm `--hf_subfolder` (mặc định suy từ tên file `*_metrics.json`); push metrics
  vào `{run_name}/`. Notebook truyền `--hf_subfolder {name}` tường minh.
- `D1_scst` load checkpoint: đường dẫn local vẫn là `runs/P_pubmed_gpt2_best.pt` (không đổi);
  chỉ đường trên HF là sub-folder.

## Tiêu chí nghiệm thu
1. Sửa `EPOCHS`/`LR`/`HF_REPO`/`COMET_PROJECT` ở cell CONFIG → mọi run dùng giá trị mới,
   không phải sửa cell khác.
2. Chạy 1 nhóm (vd A1) sinh đúng các run với tham số đúng; eval khớp cờ train.
3. Repo HF có mỗi run một folder chứa best/checkpoint/history/metrics.
4. `run_pipeline.ipynb` cũ vẫn còn nguyên.
5. Số commit HF vẫn thấp (giữ cơ chế gom commit đã có).
