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
