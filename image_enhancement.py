import os
import cv2
import numpy as np

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
            low_threshold = np.argmin(np.abs(cdf - 0.05)) / 255.0
            
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

# 設定路徑 注意路徑不能有中文字
input_dir = r"C:\Users\User\Desktop\code\DATA_train\image"
output_dir = r"C:\Users\User\Desktop\code\DATA_train\Image_Enhancement"

# 執行圖像處理，只有低門檻自動調整
auto_adjust_image(input_dir, output_dir)