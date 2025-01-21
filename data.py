import os
import tarfile
import shutil
import random

# 路徑與參數
DATASET_ROOT = "DATASET/imagenet"
TRAIN_TAR_FILE = "ILSVRC2012_img_train_t3.tar"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
VAL_SPLIT_RATIO = 0.1  # 驗證集比例

# 創建資料夾
def create_dirs():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

# 解壓主壓縮檔
def extract_main_tar():
    print("正在解壓主 tar 檔案...")
    with tarfile.open(TRAIN_TAR_FILE, "r") as tar:
        tar.extractall(TRAIN_DIR)
    print("主 tar 檔案解壓完成！")

# 解壓每個類別的 tar 檔案
def extract_class_tars():
    print("正在解壓每個類別的 tar 檔案...")
    class_tar_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".tar")]

    for class_tar in class_tar_files:
        class_name = os.path.splitext(class_tar)[0]  # 類別名稱
        class_dir = os.path.join(TRAIN_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 解壓類別 tar 檔案
        class_tar_path = os.path.join(TRAIN_DIR, class_tar)
        with tarfile.open(class_tar_path, "r") as tar:
            tar.extractall(class_dir)

        # 刪除原 tar 檔案
        os.remove(class_tar_path)
        print(f"解壓完成：{class_tar}")
    print("所有類別的 tar 檔案解壓完成！")

# 分割訓練和驗證集
def split_data():
    print("正在分割訓練和驗證集...")
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]

    for cls in classes:
        class_dir = os.path.join(TRAIN_DIR, cls)
        images = os.listdir(class_dir)
        random.shuffle(images)

        # 分配驗證集圖片
        val_count = int(len(images) * VAL_SPLIT_RATIO)
        val_images = images[:val_count]
        train_images = images[val_count:]

        # 建立驗證集類別資料夾
        val_class_dir = os.path.join(VAL_DIR, cls)
        os.makedirs(val_class_dir, exist_ok=True)

        # 移動圖片到驗證集
        for img in val_images:
            shutil.move(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

        print(f"類別 {cls}：訓練圖片數量 = {len(train_images)}, 驗證圖片數量 = {len(val_images)}")
    print("訓練與驗證集分割完成！")

if __name__ == "__main__":
    create_dirs()
    extract_main_tar()
    extract_class_tars()
    split_data()
    print(f"完成！訓練集儲存在 {TRAIN_DIR}，驗證集儲存在 {VAL_DIR}。")
