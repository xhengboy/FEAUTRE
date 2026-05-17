import os
import cv2
import numpy as np
from PIL import Image

def auto_adjust_image(input_path, output_path, high_threshold=0.4, gamma=0.35):
    """
    自適應調整圖像的對比度與亮度，只自動調整低門檻值。
    
    參數:
    input_path (str): 輸入圖像資料夾路徑
    output_path (str): 輸出圖像資料夾路徑
    high_threshold (float): 固定的高門檻值
    gamma (float): 固定的 gamma 值
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        if os.path.isfile(file_path):
            # 讀取圖像
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"無法讀取圖像: {file_name}")
                continue

            # 計算圖像直方圖
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_normalized = hist.ravel() / hist.sum()
            
            # 計算累積分布函數(CDF)
            cdf = hist_normalized.cumsum()
            
            # 只自動決定低門檻值
            low_threshold = np.argmin(np.abs(cdf - 0.03)) / 255.0
            
            # 正規化圖像
            image_normalized = image / 255.0
            
            # 應用參數進行影像增強
            image_clipped = np.clip(image_normalized, low_threshold, high_threshold)
            image_rescaled = (image_clipped - low_threshold) / (high_threshold - low_threshold)
            image_gamma_corrected = np.power(image_rescaled, gamma)
            
            # 轉換回8位元格式
            image_output = (image_gamma_corrected * 255).astype(np.uint8)

            # 儲存結果
            output_file_path = os.path.join(output_path, file_name)
            cv2.imwrite(output_file_path, image_output)
            
            print(f"已處理圖像: {file_name}")
            print(f"參數設定 - 自動低門檻: {low_threshold:.3f}, 固定高門檻: {high_threshold}, Gamma: {gamma}")

# 設定路徑
input_dir  = R"C:\Users\msi-x\Desktop\1\01\0"
output_dir = R"C:\Users\msi-x\Desktop\1\01\1"

# 執行圖像處理，只有低門檻自動調整
auto_adjust_image(input_dir, output_dir)


fusion_die = R"C:\Users\msi-x\Desktop\1\01\2"

# 取得存檔路徑的最後一部分名稱
base_filename = os.path.basename(fusion_die)

# 確保儲存目錄存在
os.makedirs(fusion_die, exist_ok=True)

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
    x_images = read_image_signals(input_dir)
    y_images = read_image_signals(output_dir)

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
        save_path = os.path.join(fusion_die, output_filename)
        fused_image.save(save_path)

if __name__ == "__main__":
    process_images()
