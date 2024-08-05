from huggingface_hub import snapshot_download


def main() :
    snapshot_download(repo_id="google/gemma-7b", local_dir="model")
    return


if __name__ == "__main__":
    main()

