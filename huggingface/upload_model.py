# Upload modela i model card-a na HF Hub. Pokrece se u Colab-u, u istom
# runtime-u gde postoji wireless_autoencoder_7_4.pt (pokreni save celiju pre).
#
# Token NIJE hardkodovan: Colab levo -> ikonica kljuca (Secrets) -> dodaj
# HF_TOKEN (Write scope) i ukljuci "Notebook access".

from google.colab import userdata
from huggingface_hub import HfApi, login

HF_USERNAME = "nikolabakic"  # zameni ako je HF username drugaciji
REPO_ID = f"{HF_USERNAME}/wireless-autoencoder-7-4"

login(token=userdata.get("HF_TOKEN"))
api = HfApi()

api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj="wireless_autoencoder_7_4.pt",
    path_in_repo="wireless_autoencoder_7_4.pt",
    repo_id=REPO_ID,
    repo_type="model",
)

# Model card: uploaduj README.md (prethodno ga snimi u Colab radni dir,
# npr. preko %%writefile README.md sa sadrzajem iz hf/README.md)
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"Gotovo: https://huggingface.co/{REPO_ID}")
