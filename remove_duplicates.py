def find_duplicates(root_dir,
                    hashfunc=imagehash.phash,
                    hash_size=8,
                    phash_threshold=5,
                    ssim_threshold=0.95):
    """
    Traverse `root_dir`, identify image pairs that are perceptually similar by pHash
    and visually similar by SSIM.

    Args:
        root_dir (str): Path to directory to search for duplicates.
        hashfunc (callable): imagehash function (phash, ahash, etc.).
        hash_size (int): Hash size for pHash hashing precision.
        phash_threshold (int): Max Hamming distance for pHash similarity.
        ssim_threshold (float): Min SSIM score (0-1) for structural similarity.

    Returns:
        list of tuples: [(path_i, path_j, ph_dist, ssim_score), ...]
    """
    records = []
    # Compute perceptual hashes
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue
            path = os.path.join(dirpath, fname)
            try:
                img = Image.open(path)
                h = hashfunc(img, hash_size=hash_size)
                records.append((path, h))
            except Exception as e:
                print(f"⚠️ Could not hash {path}: {e}")

    # Compare hashes and confirm with SSIM
    duplicates = []
    n = len(records)
    for i in range(n):
        path_i, hash_i = records[i]
        for j in range(i+1, n):
            path_j, hash_j = records[j]
            ph_dist = hash_i - hash_j
            if ph_dist <= phash_threshold:
                img_i = cv2.imread(path_i, cv2.IMREAD_GRAYSCALE)
                img_j = cv2.imread(path_j, cv2.IMREAD_GRAYSCALE)
                if img_i is None or img_j is None:
                    continue
                # resize to smallest common shape
                h_i, w_i = img_i.shape
                h_j, w_j = img_j.shape
                mh, mw = min(h_i, h_j), min(w_i, w_j)
                img_i_rs = cv2.resize(img_i, (mw, mh), interpolation=cv2.INTER_AREA)
                img_j_rs = cv2.resize(img_j, (mw, mh), interpolation=cv2.INTER_AREA)
                score = ssim(img_i_rs, img_j_rs)
                if score >= ssim_threshold:
                    duplicates.append((path_i, path_j, ph_dist, score))

    return duplicates


def remove_duplicates(duplicates, root_dir):
    """
    Remove one image from each duplicate pair.
    If one is in train and one in test, remove from train.
    Otherwise remove the second path in the pair.
    Deleted files are moved to `root_dir/deleted_duplicates`.
    """
    deleted_dir = os.path.join(root_dir, 'deleted_duplicates')
    os.makedirs(deleted_dir, exist_ok=True)
    removed = []

    for path_i, path_j, _, _ in duplicates:
        # determine train vs test
        segments_i = os.path.normpath(path_i).split(os.sep)
        segments_j = os.path.normpath(path_j).split(os.sep)
        in_train_i = 'train' in segments_i
        in_train_j = 'train' in segments_j

        # choose which to remove
        if in_train_i and not in_train_j:
            to_remove = path_i
        elif in_train_j and not in_train_i:
            to_remove = path_j
        else:
            # default: remove second
            to_remove = path_j

        # move file
        try:
            dest = os.path.join(deleted_dir, os.path.basename(to_remove))
            # avoid overwriting
            base, ext = os.path.splitext(dest)
            count = 1
            while os.path.exists(dest):
                dest = f"{base}_{count}{ext}"
                count += 1
            shutil.move(to_remove, dest)
            removed.append(to_remove)
            print(f"🗑️ Moved duplicate '{to_remove}' -> '{dest}'")
        except Exception as e:
            print(f"⚠️ Failed to remove {to_remove}: {e}")

    print(f"Summary: Moved {len(removed)} duplicate files to '{deleted_dir}'.")
    return removed


if __name__ == '__main__':
    root_folder = 'classification_task'
    print("🔍 Finding duplicates...")
    duplicates = find_duplicates(root_folder)
    if not duplicates:
        print("✅ No duplicates found.")
    else:
        print(f"⚠️ Found {len(duplicates)} duplicate pairs. Proceeding to remove duplicates...")
        removed = remove_duplicates(duplicates, root_folder)
        if removed:
            print(f"✅ Completed. {len(removed)} images removed and saved under 'deleted_duplicates'.")
        else:
            print("⚠️ No files were removed.")



