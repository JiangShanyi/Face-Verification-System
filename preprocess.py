import argparse
from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms, datasets
from facenet_pytorch import MTCNN
from tqdm import tqdm


# ------------------------ Config ------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------ Utils ------------------------

def find_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def tensor_to_pil_safe(x: torch.Tensor) -> Image.Image:
    """
    Robustly convert an image tensor returned by MTCNN to a proper PIL image.

    Handles common ranges:
      - float [0, 255]   -> scales to [0, 1]
      - float [-1, 1]    -> rescales to [0, 1]
      - float [0, 1]     -> clipped
      - uint8 [0, 255]   -> handled by to_pil_image
    """
    t = x.detach().cpu()
    if t.dtype.is_floating_point:
        tmin, tmax = float(t.min()), float(t.max())
        if tmax > 1.5:            # likely [0,255] float
            t = (t / 255.0).clamp(0.0, 1.0)
        elif tmin < -0.5:         # likely [-1,1]
            t = ((t + 1.0) / 2.0).clamp(0.0, 1.0)
        else:                     # [0,1]
            t = t.clamp(0.0, 1.0)
        return to_pil_image(t)
    else:
        return to_pil_image(t)

def build_mtcnn(
    device: torch.device,
    image_size: int = 224,
    margin: int = 20,
    min_face_size: int = 40,
    keep_all: bool = False,
) -> MTCNN:
    # post_process=False ⇒ no whitening; just aligned RGB crop
    return MTCNN(
        image_size=image_size,
        margin=margin,
        min_face_size=min_face_size,
        keep_all=keep_all,
        select_largest=not keep_all,  # if not keeping all, pick the largest
        post_process=False,
        device=device,
    )


# ------------------------ Core processing ------------------------

def save_face_obj(
    face_obj: Union[torch.Tensor, Image.Image],
    out_path: Path
):
    """
    Save a single face (tensor or PIL) to out_path (.jpg).
    """
    ensure_parent(out_path)
    if isinstance(face_obj, Image.Image):
        face_pil = face_obj.convert("RGB")
    else:
        face_pil = tensor_to_pil_safe(face_obj)
    face_pil.save(out_path.with_suffix(".jpg"), quality=95, optimize=True)

def process_one(
    img_path: Path,
    out_base: Path,
    mtcnn: MTCNN,
    keep_all: bool = False
) -> Tuple[int, float]:
    """
    Detect+align on a single image.

    Returns:
        n_saved: number of crops saved (0 if none)
        best_prob: highest detection probability (0 if none)
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return 0, 0.0

    with torch.inference_mode():
        faces, probs = mtcnn(img, return_prob=True)

    if faces is None:
        return 0, 0.0

    if keep_all:
        n_saved = 0
        best_prob = 0.0
        # faces: list[Tensor] | Tensor with batch dim
        if isinstance(faces, torch.Tensor) and faces.dim() == 4:
            faces_iter = list(faces)
        elif isinstance(faces, list):
            faces_iter = faces
        else:
            faces_iter = [faces]

        if isinstance(probs, (list, tuple, torch.Tensor)):
            probs_list = [float(p) for p in probs]
        else:
            probs_list = [float(probs)]

        for idx, face in enumerate(faces_iter):
            out_path = out_base.parent / f"{out_base.stem}_f{idx:02d}{out_base.suffix}"
            save_face_obj(face, out_path)
            n_saved += 1
        if probs_list:
            best_prob = max(probs_list)
        return n_saved, best_prob
    else:
        # Single best face
        save_face_obj(faces, out_base)
        best_prob = float(probs if probs is not None else 0.0)
        return 1, best_prob


def preprocess_folder(
    in_root: Path,
    out_root: Path,
    image_size: int = 224,
    margin: int = 20,
    min_face_size: int = 40,
    keep_all: bool = False,
    skip_existing: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = build_mtcnn(
        device, image_size=image_size, margin=margin,
        min_face_size=min_face_size, keep_all=keep_all
    )

    all_imgs = find_images(in_root)
    if not all_imgs:
        print(f"[WARN] No images found under: {in_root}")
        return

    saved_total, failed = 0, 0
    for src in tqdm(all_imgs, desc="Detect+Align (MTCNN)", unit="img"):
        rel = src.relative_to(in_root)
        dst = (out_root / rel).with_suffix(".jpg")

        if skip_existing and dst.exists() and not keep_all:
            saved_total += 1
            continue

        n_saved, _ = process_one(src, dst, mtcnn, keep_all=keep_all)
        if n_saved > 0:
            saved_total += n_saved
        else:
            failed += 1

    print(f"\nDone. Saved: {saved_total},  No-face/failed: {failed},  Out: {out_root}")


# ------------------------ Dataloader (Normalization) ------------------------

def make_aligned_dataloader(
    root_aligned: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True
):
    """
    Loads aligned 224×224 crops and applies ImageNet normalization.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(str(root_aligned), transform=transform)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True
    )
    return loader, ds.classes


# ------------------------ CLI ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect & align faces → 224×224 → save crops. "
                    "Normalization is applied later in the dataloader."
    )
    parser.add_argument("--in", dest="in_root", required=True, type=Path,
                        help="Input folder (e.g., data/raw). Can contain class/ID subfolders.")
    parser.add_argument("--out", dest="out_root", required=True, type=Path,
                        help="Output folder for aligned crops (e.g., data/aligned).")
    parser.add_argument("--size", type=int, default=224, help="Output size (square). Default: 224")
    parser.add_argument("--margin", type=int, default=20, help="Extra margin around the face box.")
    parser.add_argument("--min-face", type=int, default=40, help="Minimum face size.")
    parser.add_argument("--keep-all", action="store_true",
                        help="Save all detected faces per image (appends _f00, _f01...).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild all (ignore existing files).")
    args = parser.parse_args()

    preprocess_folder(
        in_root=args.in_root,
        out_root=args.out_root,
        image_size=args.size,
        margin=args.margin,
        min_face_size=args.min_face,
        keep_all=args.keep_all,
        skip_existing=not args.rebuild
    )

    print("\n[Tip] Training with normalization:")
    print("  from prep_faces import make_aligned_dataloader")
    print(f"  loader, classes = make_aligned_dataloader(Path('{args.out_root}'))")


if __name__ == "__main__":
    main()
