"""
dataset.py — Data loading for CRNN Handwritten Character Recognition
=====================================================================
Supports two datasets out of the box:

  1. EMNIST (auto-downloaded via torchvision) — character-level
     62 classes: digits 0-9, uppercase A-Z, lowercase a-z
     Source: https://www.nist.gov/itl/products-and-services/emnist-dataset

  2. IAM Handwriting Database — word/line level
     Must be downloaded manually from:
     https://fki.inf.unibe.ch/databases/iam-handwriting-database
     Then pass --dataset iam --iam_root /path/to/iam to train.py

Quick check:
    python dataset.py --dataset emnist        # auto-downloads ~560 MB
    python dataset.py --dataset iam --iam_root ./data/iam
"""

import os
import argparse
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as tvd
from tqdm import tqdm


# ── Vocabulary ────────────────────────────────────────────────────────────────
# ── Vocabulary ────────────────────────────────────────────────────────────────
# Combined Vocabulary: 
# 1. Digits (0-9)
# 2. Uppercase (A-Z)
# 3. Lowercase (a-z)
# 4. Hindi (Devanagari + Matras + Symbols)
ASCII_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHARS       = ASCII_CHARS
BLANK_IDX   = 0
CHAR2IDX    = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR    = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # 62 (ASCII) + 1 (Blank) = 63


def encode(text: str):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def decode_ctc(indices) -> str:
    """Greedy CTC: collapse repeats, strip blanks."""
    result, prev = [], None
    for idx in indices:
        idx = int(idx)
        if idx != prev and idx != BLANK_IDX:
            result.append(IDX2CHAR.get(idx, "?"))
        prev = idx
    return "".join(result)


# ── Transforms ────────────────────────────────────────────────────────────────
def make_transform(train: bool, h: int, w: int):
    ops = [T.Grayscale(), T.Resize((h, w))]
    if train:
        ops += [
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), shear=5),
            T.ColorJitter(brightness=0.35, contrast=0.35),
            T.GaussianBlur(3, sigma=(0.1, 1.2)),
        ]
    ops += [T.ToTensor(), T.Normalize([0.5], [0.5])]
    return T.Compose(ops)


def collate_fn(batch):
    images, labels, lengths, texts = zip(*batch)
    return (
        torch.stack(images, 0),
        torch.cat(labels),
        torch.tensor(lengths, dtype=torch.long),
        list(texts),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1.  EMNIST  (auto-download)
# ══════════════════════════════════════════════════════════════════════════════
class EMNISTDataset(Dataset):
    """
    Wraps torchvision EMNIST 'byclass' split.
    torchvision downloads the data automatically (~560 MB, first run only).

    Each sample = single character image → single character string.
    img_h and img_w are usually kept at 32×32 for characters.
    """

    # EMNIST ByClass label index → character
    _EMNIST_CHARS = list(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )

    def __init__(self, root="./data/emnist", train=True,
                 img_h=32, img_w=32, max_samples=None):
        self.tfm = make_transform(train, img_h, img_w)
        split_name = "train" if train else "test"
        print(f"[EMNIST] Loading {split_name} split → {root}")

        self.base = tvd.EMNIST(
            root=root, split="byclass",
            train=train, download=True, transform=None,
        )

        n = len(self.base)
        if max_samples and max_samples < n:
            idx = torch.randperm(n)[:max_samples].tolist()
        else:
            idx = list(range(n))
        self.idx = idx
        print(f"[EMNIST] {len(self.idx):,} samples ready.")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img_pil, label_int = self.base[self.idx[i]]
        if not isinstance(img_pil, Image.Image):
            img_pil = T.ToPILImage()(img_pil)
        # EMNIST letters are stored transposed — fix it
        img_pil = img_pil.transpose(Image.TRANSPOSE)

        img = self.tfm(img_pil)
        char = self._EMNIST_CHARS[label_int] if label_int < len(self._EMNIST_CHARS) else "0"
        enc = encode(char) or [1]
        return img, torch.tensor(enc, dtype=torch.long), len(enc), char


def get_emnist_loaders(root="./data/emnist",
                       img_h=32, img_w=32,
                       batch_size=128, num_workers=4,
                       max_train=None, max_val=None):
    """
    Returns (train_loader, val_loader).
    EMNIST is downloaded automatically on the first call.
    """
    tr = EMNISTDataset(root, train=True,  img_h=img_h, img_w=img_w, max_samples=max_train)
    va = EMNISTDataset(root, train=False, img_h=img_h, img_w=img_w, max_samples=max_val)

    pin = torch.cuda.is_available()   # pin_memory only works on CUDA, not MPS/CPU
    kw  = dict(collate_fn=collate_fn, pin_memory=pin, num_workers=num_workers)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,  drop_last=True, **kw),
        DataLoader(va, batch_size=batch_size, shuffle=False, **kw),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Hindi DHCD  (Devanagari Handwritten Character Dataset)
# ══════════════════════════════════════════════════════════════════════════════
class HindiDHCDDataset(Dataset):
    """
    Loads Devanagari Handwritten Character Dataset (DHCD).
    Structure: ./data/hindi/DevanagariHandwrittenCharacterDataset/Train/character_1_ka/...
    """
    def __init__(self, root="./data/hindi", train=True, img_h=32, img_w=32, max_samples=None):
        self.tfm = make_transform(train, img_h, img_w)
        split = "Train" if train else "Test"
        base_dir = os.path.join(root, "DevanagariHandwrittenCharacterDataset", split)
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Hindi dataset not found at {base_dir}. Run download_hindi_data.py first.")
            
        self.samples = []
        # Mapping DHCD folder names to our HINDI_CHARS vocabulary
        # Folder names are 'character_1_ka', 'digit_0', etc.
        folders = sorted(os.listdir(base_dir))
        
        # We need a mapping from folder name to the actual character
        # Simplified for now: Mapping folder index to HINDI_CHARS
        # In a real scenario, we'd use a more precise mapping.
        for folder in tqdm(folders, desc=f"Loading Hindi {split}", leave=False):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path): continue
            
            # Map folder name to character (this is a simplified logic for DHCD structure)
            # Digit folders: digit_0 -> '0', etc.
            # Character folders: character_1_ka -> 'क', etc.
            char = self._map_folder_to_char(folder)
            
            files = os.listdir(folder_path)
            if max_samples:
                files = files[:max_samples // len(folders)]
                
            for f in files:
                self.samples.append((os.path.join(folder_path, f), char))
                
        print(f"[Hindi] {split}: {len(self.samples):,} samples loaded.")

    def _map_folder_to_char(self, folder_name):
        # DHCD Folder Mapping (VGG order)
        # Consonants 1-36, Digits 0-9
        mapping = {
            "character_1_ka": "क", "character_2_kha": "ख", "character_3_ga": "ग", "character_4_gha": "घ", "character_5_kna": "ङ",
            "character_6_cha": "च", "character_7_chha": "छ", "character_8_ja": "ज", "character_9_jha": "झ", "character_10_yna": "ञ",
            "character_11_taamatar": "ट", "character_12_thaa": "ठ", "character_13_daa": "ड", "character_14_dhaa": "ढ", "character_15_adna": "ण",
            "character_16_tabala": "त", "character_17_tha": "थ", "character_18_da": "द", "character_19_dha": "ध", "character_20_na": "न",
            "character_21_pa": "प", "character_22_pha": "फ", "character_23_ba": "ब", "character_24_bha": "भ", "character_25_ma": "म",
            "character_26_yaw": "य", "character_27_ra": "र", "character_28_la": "ल", "character_29_waw": "व", "character_30_mirdhak_sha": "ष",
            "character_31_petchiryak_sha": "ष", "character_32_patalos_sa": "स", "character_33_ha": "ह", "character_34_chhya": "क्ष", "character_35_tra": "त्र", "character_36_gya": "ज्ञ",
            "digit_0": "0", "digit_1": "1", "digit_2": "2", "digit_3": "3", "digit_4": "4", "digit_5": "5", "digit_6": "6", "digit_7": "7", "digit_8": "8", "digit_9": "9"
        }
        return mapping.get(folder_name, "?")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, char = self.samples[i]
        img = Image.open(img_path).convert("L")
        img = self.tfm(img)
        enc = encode(char) or [1]
        return img, torch.tensor(enc, dtype=torch.long), len(enc), char

def get_hindi_loaders(root="./data/hindi", img_h=32, img_w=32, batch_size=128, num_workers=2):
    tr = HindiDHCDDataset(root, train=True, img_h=img_h, img_w=img_w)
    va = HindiDHCDDataset(root, train=False, img_h=img_h, img_w=img_w)
    kw = dict(collate_fn=collate_fn, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return DataLoader(tr, batch_size=batch_size, shuffle=True, **kw), DataLoader(va, batch_size=batch_size, shuffle=False, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  IAM Handwriting Database  (manual download required)
# ══════════════════════════════════════════════════════════════════════════════
class IAMDataset(Dataset):
    """
    IAM word-level handwriting recognition.

    Download steps
    --------------
    1. Register at https://fki.inf.unibe.ch/databases/iam-handwriting-database
    2. Download:
         words.tgz         → extract to <iam_root>/words/
         ascii.tgz         → extract to <iam_root>/ascii/
    3. Run train.py with --dataset iam --iam_root <iam_root>

    Directory layout expected
    -------------------------
    <iam_root>/
        words/
            a01/
                a01-000u/
                    a01-000u-00-00.png
                    ...
        ascii/
            words.txt
    """

    # Standard IAM split by form prefix
    _SPLITS = {
        "train": {
            "a01","a02","a03","a04","a05","a06","a07","a08",
            "b01","b02","b03","b04","b05","b06",
            "c01","c02","d01","d02","e01","f01","g01","h01",
        },
        "val":  {"c03","c04","c05","c06"},
        "test": {"a09","b07","c07","d05","d06","d07","e02","f02","g02","h02"},
    }

    def __init__(self, iam_root, split="train", img_h=32, img_w=128):
        self.tfm = make_transform(split == "train", img_h, img_w)
        self.samples = []
        self._parse(iam_root, split)
        print(f"[IAM] {split}: {len(self.samples):,} samples")

    def _parse(self, root, split):
        words_txt = os.path.join(root, "ascii", "words.txt")
        if not os.path.exists(words_txt):
            raise FileNotFoundError(
                f"\nwords.txt not found at:\n  {words_txt}\n\n"
                "Download IAM from:\n"
                "  https://fki.inf.unibe.ch/databases/iam-handwriting-database\n"
                "Extract ascii.tgz → <iam_root>/ascii/\n"
                "Extract words.tgz → <iam_root>/words/\n"
            )
        allowed = self._SPLITS.get(split, self._SPLITS["train"])

        with open(words_txt, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.strip().split()
                if len(p) < 9 or p[1] != "ok":
                    continue
                wid = p[0]
                if wid.split("-")[0] not in allowed:
                    continue
                text = p[8]
                enc  = encode(text)
                if not enc:
                    continue
                segs = wid.split("-")
                img_path = os.path.join(
                    root, "words",
                    segs[0], "-".join(segs[:2]), wid + ".png"
                )
                if os.path.exists(img_path):
                    self.samples.append((img_path, text, enc))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, text, enc = self.samples[i]
        try:
            img = Image.open(img_path).convert("L")
            img = ImageOps.autocontrast(img)
        except Exception:
            img = Image.new("L", (128, 32), 255)
        img = self.tfm(img)
        return img, torch.tensor(enc, dtype=torch.long), len(enc), text


def get_iam_loaders(iam_root, img_h=32, img_w=128,
                    batch_size=32, num_workers=4):
    tr = IAMDataset(iam_root, "train", img_h, img_w)
    va = IAMDataset(iam_root, "val",   img_h, img_w)
    pin = torch.cuda.is_available()
    kw  = dict(collate_fn=collate_fn, pin_memory=pin, num_workers=num_workers)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,  drop_last=True, **kw),
        DataLoader(va, batch_size=batch_size, shuffle=False, **kw),
    )


# ── Inference helper ──────────────────────────────────────────────────────────
def preprocess_for_inference(image_path_or_pil, img_h=32, img_w=128):
    """Return (1,1,H,W) tensor ready for model.forward()."""
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("L")
    else:
        img = image_path_or_pil.convert("L")
    img = ImageOps.autocontrast(img)
    return make_transform(False, img_h, img_w)(img).unsqueeze(0)


# ── Quick CLI sanity check ────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    choices=["emnist", "iam"], default="emnist")
    ap.add_argument("--iam_root",   default="./data/iam")
    ap.add_argument("--data_root",  default="./data/emnist")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if args.dataset == "emnist":
        tr_dl, va_dl = get_emnist_loaders(
            args.data_root, batch_size=args.batch_size,
            max_train=4000, max_val=800
        )
    else:
        tr_dl, va_dl = get_iam_loaders(args.iam_root, batch_size=args.batch_size)

    imgs, labels, lengths, texts = next(iter(tr_dl))
    print(f"Batch images  : {imgs.shape}")
    print(f"Labels (cat)  : {labels.shape}   lengths={lengths[:8].tolist()}")
    print(f"Sample texts  : {texts[:8]}")
    print("dataset.py check ✓")
