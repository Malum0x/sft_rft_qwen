from huggingface_hub import snapshot_download

snapshot_download(repo_id="NousResearch/RLVR_Coding_Problems", repo_type="dataset", local_dir=".")
