from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.guided_backprop import GuidedBackpropReLUModel




DATA_ROOT = '/content/drive/MyDrive/classification_task'
SAVE_DIR = '/content/drive/MyDrive/Convnext_small'
CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']


class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.metadata = []
        if not os.path.exists(root_dir):
            print(f"❌ Directory not found: {root_dir}")
            return
        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"⚠️ Class directory missing: {class_dir}")
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
                    path = os.path.join(class_dir, filename)
                    if not os.path.isfile(path):
                        continue
                    tumor_type, plane = extract_metadata_from_filename(filename)
                    if tumor_type and plane:
                        self.file_paths.append(path)
                        self.labels.append(CLASS_TO_IDX[tumor_type])
                        self.metadata.append((tumor_type, plane))
        print(f"Loaded {len(self.file_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 255)
        if self.transform:
            # Pass image as a named argument for Albumentations
            image = self.transform(image=np.array(image))['image']
        return image, self.labels[idx]

def create_model():
    from torchvision.models import convnext_small, ConvNeXt_Small_Weights
    model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    num_ftrs = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(num_ftrs, eps=1e-6),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    return model

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


def denormalize(img: torch.Tensor, mean, std):
    """img: (C,H,W) tensor, normalized. Returns de-normalized tensor (C,H,W)."""
    mean = torch.tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.tensor(std, dtype=img.dtype, device=img.device)
    return img * std[:, None, None] + mean[:, None, None]

def _normalize_map(arr: np.ndarray, eps=1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() + eps)
    return arr

def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """
    Accepts (H,W), (C,H,W), (H,W,C) or (1,C,H,W)->(C,H,W).
    Returns float32 (H,W,C) with 3 channels.
    """
    arr = np.asarray(arr)
    # Strip leading batch if present
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        hwc = np.stack([arr] * 3, axis=-1)
        return hwc.astype(np.float32)

    if arr.ndim == 3:
        # Case A: (H,W,C)
        if arr.shape[-1] in (1, 3) and arr.shape[0] not in (1, 3):
            hwc = arr
            if hwc.shape[-1] == 1:
                hwc = np.repeat(hwc, 3, axis=-1)
            return hwc.astype(np.float32)

        # Case B: (C,H,W)
        if arr.shape[0] in (1, 3):
            c, h, w = arr.shape
            chw = arr
            if c == 1:
                chw = np.repeat(chw, 3, axis=0)
            return chw.transpose(1, 2, 0).astype(np.float32)

        # Ambiguous fallback: if middle two dims equal, treat as CHW
        if arr.shape[1] == arr.shape[2]:
            chw = arr[:3] if arr.shape[0] > 3 else arr
            return chw.transpose(1, 2, 0).astype(np.float32)

        # Otherwise assume HWC
        hwc = arr[..., :3] if arr.shape[-1] > 3 else arr
        if hwc.shape[-1] == 1:
            hwc = np.repeat(hwc, 3, axis=-1)
        return hwc.astype(np.float32)

    raise ValueError(f"_to_hwc: unexpected shape {arr.shape}")

def _get_stage_block(model: nn.Module, idx: int):
    """Adapts to models with model.features list of stages; uses dwconv if present."""
    stage = model.features[idx]
    block = stage[-1] if isinstance(stage, nn.Sequential) else stage
    return getattr(block, "dwconv", block)


# -----------------------------
# Main visualization function
# -----------------------------

def apply_gradcams(
    model: nn.Module,
    test_dataset,
    device: torch.device,
    mean,
    std,
    save_dir: str,
    num_images: int = 200,
    target_layer_indices=(3, 5, 7),
    use_pred_as_target: bool = False,
    dpi: int = 160,
):
    """
    Saves panels with: Original | Grad-CAM | Grad-CAM++ | Overlay (Grad-CAM) | Guided Grad-CAM

    - Guided Grad-CAM = GuidedBackprop ⨉ Grad-CAM (original definition).
    - Set `use_pred_as_target=True` to target top-1 prediction instead of ground-truth.
    """
    out_dir = os.path.join(save_dir, "Gradcams")
    os.makedirs(out_dir, exist_ok=True)

    # Resolve target layers
    target_layers = []
    for i in target_layer_indices:
        try:
            target_layers.append(_get_stage_block(model, i))
        except Exception as e:
            raise RuntimeError(
                f"Couldn't resolve target layer at model.features[{i}]. "
                "Adjust `target_layer_indices` for your backbone."
            ) from e

    # Handle odd activations that aren't 4D
    def reshape_transform(t):
        return t if t.ndim == 4 else t[:, :, None, None]

    # CAM objects
    cam_std = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    cam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # Guided backprop
    gb_model = GuidedBackpropReLUModel(model=model, device=device)

    model.eval()
    torch.set_grad_enabled(True)

    random.seed()
    indices = random.sample(range(len(test_dataset)), min(num_images, len(test_dataset)))
    file_paths = getattr(test_dataset, "file_paths", None)

    for i, idx in enumerate(indices, 1):
        try:
            # -------------------------
            # Load one sample
            # -------------------------
            sample = test_dataset[idx]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                img, label = sample[0], sample[1]
            else:
                # If dataset returns only image, pretend label 0
                img, label = sample, 0

            if img is None:
                continue

            inp = img.unsqueeze(0).to(device)
            inp.requires_grad_(True)

            # Forward once for prediction
            with torch.no_grad():
                out = model(inp)
                probs = torch.softmax(out, dim=1)[0]
                pred = int(probs.argmax().item())

            target_class = pred if use_pred_as_target else int(label)
            targets = [ClassifierOutputTarget(target_class)]

            # -------------------------
            # CAMs
            # -------------------------
            cam_maps_gc = cam_std(input_tensor=inp, targets=targets)   # (N,H,W)
            cam_maps_pp = cam_pp(input_tensor=inp, targets=targets)

            heatmap_gc = _normalize_map(np.mean(cam_maps_gc, axis=0))
            heatmap_gcp = _normalize_map(np.mean(cam_maps_pp, axis=0))

            # -------------------------
            # Guided backprop
            # -------------------------
            raw_gb = gb_model(inp, target_category=target_class)
            if isinstance(raw_gb, list):
                raw_gb = raw_gb[0]
            if torch.is_tensor(raw_gb):
                raw_gb = raw_gb.detach().cpu().numpy()

            gb = _to_hwc(raw_gb)
            gb = _normalize_map(gb)

            # Sanity checks (helpful to keep!)
            assert heatmap_gc.ndim == 2, f"heatmap_gc shape {heatmap_gc.shape}"
            assert gb.ndim == 3 and gb.shape[-1] == 3, f"guided backprop HWC shape {gb.shape}"
            assert gb.shape[:2] == heatmap_gc.shape, f"gb HW {gb.shape[:2]} vs cam {heatmap_gc.shape}"

            # Guided Grad-CAM uses plain Grad-CAM heatmap
            guided = gb * heatmap_gc[:, :, None]
            guided = _normalize_map(guided)

            # -------------------------
            # Build visuals
            # -------------------------
            denorm = denormalize(img, mean, std).detach().cpu()
            rgb = _to_hwc(denorm.numpy())
            rgb = np.clip(rgb, 0, 1).astype(np.float32)

            overlay = show_cam_on_image(rgb.astype(np.float32), heatmap_gc, use_rgb=True, image_weight=0.7)

            # File name
            name = None
            if file_paths is not None and idx < len(file_paths):
                path = file_paths[idx]
                name = os.path.basename(path) if path else None
            if not name:
                name = f"idx_{idx}"
            root = os.path.splitext(name)[0]

            # Titles
            true_str = (
                f"True: {CLASSES[label]}"
                if "CLASSES" in globals() and isinstance(label, (int, np.integer)) and label < len(CLASSES)
                else f"True: {label}"
            )
            pred_str = (
                f"Pred: {CLASSES[pred]} ({probs[pred]:.2f})"
                if "CLASSES" in globals() and pred < len(CLASSES)
                else f"Pred: {pred} ({probs[pred]:.2f})"
            )

            # Plot
            fig, ax = plt.subplots(1, 5, figsize=(25, 5), dpi=dpi)
            ax[0].imshow(rgb);            ax[0].set_title(f"Original\n{true_str}");           ax[0].axis("off")
            ax[1].imshow(heatmap_gc, cmap="jet");  ax[1].set_title("Grad-CAM");                ax[1].axis("off")
            ax[2].imshow(heatmap_gcp, cmap="jet"); ax[2].set_title("Grad-CAM++");              ax[2].axis("off")
            ax[3].imshow(overlay);         ax[3].set_title(f"Overlay (Grad-CAM)\n{pred_str}"); ax[3].axis("off")
            ax[4].imshow(guided);          ax[4].set_title("Guided Grad-CAM");                 ax[4].axis("off")

            fig.suptitle(name, y=1.04, fontsize=12)
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"{root}_gradcams.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

            print(f"[{i}/{len(indices)}] saved {os.path.basename(out_path)}")

        except AssertionError as e:
            print(f"[{i}/{len(indices)}] index {idx}: shape mismatch -> {e}")
        except Exception as e:
            print(f"[{i}/{len(indices)}] index {idx}: skipped due to error -> {repr(e)}")

    print(f"All visuals saved to {out_dir}")


if __name__ == "__main__":
    # Load config to get normalization parameters
    config_path = os.path.join(SAVE_DIR, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get mean and std from config
    mean = config['normalization_mean']
    std = config['normalization_std']

    # Create model and load pre-trained weights
    model = create_model().to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt'), map_location=torch.device('cpu')))
    model.eval()

    val_tf = A.Compose([ A.SmallestMaxSize(max_size=256, p=1.0), A.CenterCrop(height=256, width=256, p=1.0), A.Normalize(mean=mean, std=std, p=1.0), ToTensorV2(p=1.0) ])


    test_dataset = ImageClassificationDataset(
        os.path.join(DATA_ROOT, 'test'),
        transform=val_tf
    )

    # Apply Grad-CAM
    apply_gradcams(
        model=model,
        test_dataset=test_dataset,
        device=device,
        mean=mean,
        std=std,
        save_dir=SAVE_DIR,
        num_images=200  # Number of images to visualize
    )






