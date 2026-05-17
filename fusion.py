import os
import numpy as np
from PIL import Image

# 資料夾路徑 注意路徑不能有中文字
x_dir    = R"C:\Users\User\Desktop\code\DATA_train\image"
y_dir    = R"C:\Users\User\Desktop\code\DATA_train\Image_Enhancement"
save_dir = R"C:\Users\User\Desktop\code\DATA_train\Image_Enhancement_fusion"


# 取得存檔路徑的最後一部分名稱
base_filename = os.path.basename(save_dir)

# 確保儲存目錄存在
os.makedirs(save_dir, exist_ok=True)

# 讀取圖檔，並轉換為灰階信號
def read_image_signals(directory):
    signals = []
    
    # 提取檔案名稱中的數字進行排序
    def extract_number(filename):
        # 提取檔名中的數字，沒有數字則返回浮點無窮大（排在最後）
        import re
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')
    
    for filename in sorted(os.listdir(directory), key=extract_number):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 打開圖檔並轉為灰階
            image = Image.open(filepath).convert("L")
            signals.append(np.array(image))
    
    if not signals:
        raise ValueError(f"資料夾 {directory} 為空或無有效圖檔，無法處理信號。")
    
    return signals


# 計算峭度指數 Ki (公式 4)
def kurtosis_index(signal):
    mean_val = np.mean(signal)
    variance = np.mean((signal - mean_val)**2)
    return np.mean((signal - mean_val)**4) / (variance**2)

# 計算交叉相關 Ri,j (公式 1)
def cross_correlation(x_i, x_j):
    N = x_i.size
    return np.sum(x_i * x_j) / N

# 計算 E_i,j 和 E_i (公式 2 和 3)
def cross_correlation_energy(x_signals):
    k = len(x_signals)
    E_matrix = np.zeros((k, k))
    E_total = np.zeros(k)
    for i in range(k):
        for j in range(k):
            if i != j:
                E_matrix[i, j] = cross_correlation(x_signals[i].flatten(), x_signals[j].flatten())**2
        E_total[i] = np.sqrt(np.sum(E_matrix[i]))
    return E_matrix, E_total

# 加權融合信號 (公式 5 和 6)
def weighted_fusion(x_signals, E_total, kurtosis_indices):
    weights = E_total * kurtosis_indices
    weights /= np.sum(weights)
    fused_signal = np.sum([weights[i] * x_signals[i] for i in range(len(x_signals))], axis=0)
    return fused_signal

# 主要處理函數
def process_images():
    # 讀取 X 和 Y 資料夾的圖像信號
    x_images = read_image_signals(x_dir)
    y_images = read_image_signals(y_dir)

    if len(x_images) != len(y_images):
        raise ValueError("X 資料夾和 Y 資料夾中的圖像數量不一致。")

    for idx, (x_image, y_image) in enumerate(zip(x_images, y_images)):
        if x_image.shape != y_image.shape:
            raise ValueError(f"圖像對 {idx + 1} 的尺寸不一致，無法融合。")

        # 計算交叉相關能量和峭度指數
        E_matrix, E_total = cross_correlation_energy([x_image, y_image])
        kurtosis_indices = np.array([kurtosis_index(x_image), kurtosis_index(y_image)])

        # 加權融合信號
        fused_signal = weighted_fusion([x_image, y_image], E_total, kurtosis_indices)

        # 將信號轉為灰階圖像
        fused_image = Image.fromarray(fused_signal.astype(np.uint8))

        # 儲存結果
        #output_filename = f"{base_filename}_{idx + 1}.jpg"
        output_filename = f"{idx + 1}.jpg"
        save_path = os.path.join(save_dir, output_filename)
        fused_image.save(save_path)

if __name__ == "__main__":
    process_images()
