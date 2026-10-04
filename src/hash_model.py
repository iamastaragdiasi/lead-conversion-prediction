# Writes or verifies a SHA-256 hash of the saved model.
#   python src/hash_model.py          -> writes models/lead_conversion_pipeline.joblib.sha256
#   python src/hash_model.py --check  -> exits 1 if the model file was changed
import hashlib
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "models" / "lead_conversion_pipeline.joblib"
HASH_FILE = MODEL.with_name(MODEL.name + ".sha256")


def digest():
    h = hashlib.sha256()
    with MODEL.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = HASH_FILE.read_text().split()[0] == digest()
        print("Model hash OK" if ok else "MODEL HASH MISMATCH: do not load this file")
        sys.exit(0 if ok else 1)
    HASH_FILE.write_text(f"{digest()}  {MODEL.name}\n")
    print(f"Wrote {HASH_FILE}")
