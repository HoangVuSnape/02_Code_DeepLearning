# src/integrations/hf_push.py
"""Push file len HuggingFace Hub. repo_id rong / thieu token -> no-op."""
import os


def push_file(repo_id, local_path, token, path_in_repo=None,
              repo_type="model", private=True, verbose=True):
    if not repo_id or not token:
        return False
    if not os.path.exists(local_path):
        if verbose:
            print(f"(hf push bo qua: khong thay file {local_path})")
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=private)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo or os.path.basename(local_path),
            repo_id=repo_id, repo_type=repo_type)
        if verbose:
            print(f"🤗 Da push {local_path} -> {repo_id}")
        return True
    except Exception as e:
        print(f"(hf push that bai: {e})")
        return False


def push_files(repo_id, files, token, repo_type="model", private=True, verbose=True):
    """Upload nhieu file voi ten dich TUY Y trong 1 COMMIT (tranh 429, dat sub-folder).

    files: list (local_path, path_in_repo). Vi du: ("runs/A1_best.pt", "A1_lstm_all/best.pt").
    Bo qua file khong ton tai. repo_id/token rong -> no-op.
    """
    if not repo_id or not token:
        return False
    existing = [(lp, pir) for lp, pir in files if lp and os.path.exists(lp)]
    if not existing:
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=private)
        ops = [CommitOperationAdd(path_in_repo=pir, path_or_fileobj=lp) for lp, pir in existing]
        api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=ops,
                          commit_message=f"Upload {len(ops)} file(s)")
        if verbose:
            print(f"🤗 Da push {len(ops)} file -> {repo_id} (1 commit)")
        return True
    except Exception as e:
        print(f"(hf push files that bai: {e})")
        return False


def push_folder(repo_id, folder, token, allow_patterns=None,
                repo_type="model", private=True, verbose=True):
    """Upload NHIEU file trong 1 COMMIT duy nhat (tranh 429 rate limit commit).

    allow_patterns: list ten/pattern file can day (vd ["A1_best.pt", "*.json"]).
    None -> day toan bo folder. repo_id/token rong -> no-op.
    """
    if not repo_id or not token:
        return False
    if not os.path.isdir(folder):
        if verbose:
            print(f"(hf push bo qua: khong thay thu muc {folder})")
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=private)
        api.upload_folder(
            folder_path=folder, repo_id=repo_id, repo_type=repo_type,
            allow_patterns=allow_patterns)
        if verbose:
            print(f"🤗 Da push {folder} ({allow_patterns or 'tat ca'}) -> {repo_id} (1 commit)")
        return True
    except Exception as e:
        print(f"(hf push folder that bai: {e})")
        return False
