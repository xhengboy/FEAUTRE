import os
from tqdm import tqdm
from PIL import Image 
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler, autocast
import cv2
import csv
import gc

# =========================== 設定區 ===========================

# 設定檔
class Config:
    # 模型和訓練配置
    CLASS_NAMES = ['Cracks', 'Hole', 'Contamination']
    NUM_CLASSES = len(CLASS_NAMES)

    # 交叉驗證設定
    K_FOLDS = 5    # K 折交叉驗證
    EPOCHS = 300     # 訓練次數
    BATCH_SIZE = 32 # 批次大小
    
    # 學習率和優化器設定
    LEARNING_RATE = 0.01
    
    # 圖像預處理
    IMAGE_SIZE = (640, 640) # 圖片大小

    # 數據路徑配置
    BASE_DIR = R"C:\Users\User\Desktop\code\DATA_train"
    IMAGE_DIR = os.path.join(BASE_DIR, "image")
    LABEL_DIR = os.path.join(BASE_DIR, "label1")
    OUTPUT_DIR = os.path.join(BASE_DIR,"640_SIZE_MobileNetV3","new","Single_fusion", f"Epochs_{EPOCHS}_BatchSize_{BATCH_SIZE}_LR_{LEARNING_RATE}")
    Accuracy = os.path.join(BASE_DIR, "640_SIZE_MobileNetV3", "new", "Single_fusion")
    # 隨機種子
    RANDOM_SEED = 42
    #影像增強
    gamma=0.35
    high_threshold=0.4

    #注意力
    #較小值（如8、16）：提供更精細的特徵表示，適合檢測細微瑕疵
    #較大值（如32、64）：減少計算量，適合大面積瑕疵檢測
    reduction_1=16 #控制CBAM
    reduction=16  #控制CA
# ===============================================================

# 自訂資料集類別 - 保持不變
class PCBDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            labels = []
            for line in f.readlines():
                class_id = float(line.strip().split()[0])
                labels.append(int(class_id))
        label = labels[0] if labels else 0
        if self.transform:
            image = self.transform(image)
        return image, label
    



class AdaptiveImageEnhancement(nn.Module):
    def __init__(self, high_threshold=Config.high_threshold, gamma=Config.gamma):
        super(AdaptiveImageEnhancement, self).__init__()
        self.high_threshold = high_threshold
        self.gamma = gamma

    def forward(self, x):
        """
        對輸入的批次圖像進行增強處理
        
        參數:
        x (torch.Tensor): 輸入圖像批次，形狀為 (batch_size, channels, height, width)
        
        返回:
        torch.Tensor: 增強後的圖像
        """
        # 將 PyTorch Tensor 轉換為 NumPy 數組，並移動到 CPU
        x_np = x.detach().cpu().numpy()
        
        # 儲存增強後的圖像
        enhanced_images = []
        
        for image in x_np:
            # 將圖像從 (C, H, W) 轉換為 (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            
            # 對每個通道單獨處理
            enhanced_channels = []
            for channel in range(image.shape[2]):
                channel_image = image[:, :, channel]
                
                # 計算直方圖
                hist = cv2.calcHist([channel_image], [0], None, [256], [0, 256])
                hist_sum = hist.sum()
                
                # 防止除以零
                if hist_sum == 0:
                    hist_normalized = np.zeros_like(hist.ravel())
                else:
                    hist_normalized = hist.ravel() / hist_sum
                
                # 計算累積分布函數(CDF)
                cdf = hist_normalized.cumsum()
                
                # 計算低門檻值
                low_threshold = np.argmin(np.abs(cdf - 0.05)) / 255.0
                
                # 正規化通道
                channel_normalized = channel_image / 255.0
                
                # 應用增強
                channel_clipped = np.clip(channel_normalized, low_threshold, self.high_threshold)
                channel_rescaled = (channel_clipped - low_threshold) / (self.high_threshold - low_threshold)
                channel_gamma_corrected = np.power(channel_rescaled, self.gamma)
                
                # 轉換回8位元格式
                channel_output = (channel_gamma_corrected * 255).astype(np.uint8)
                enhanced_channels.append(channel_output)
            
            # 重新組合通道
            enhanced_image = np.stack(enhanced_channels, axis=-1)
            
            # 轉換回 (C, H, W)
            enhanced_image = np.transpose(enhanced_image, (2, 0, 1))
            enhanced_images.append(enhanced_image)
        
        # 將 NumPy 數組轉換回 PyTorch Tensor
        enhanced_tensor = torch.from_numpy(np.array(enhanced_images)).float() / 255.0
        
        # 確保 Tensor 與輸入圖像在同一設備上
        enhanced_tensor = enhanced_tensor.to(x.device)
        
        return enhanced_tensor

#協調注意力（Coordinated Attention, CA）
class CoordinateAttention2D(nn.Module):
    """
    2D Coordinate Attention Module especially designed for defect detection
    
    特點:
    1. 分別在水平和垂直方向上編碼位置信息
    2. 捕捉長距離依賴關係
    3. 適應不同尺寸和形狀的瑕疵
    
    Parameters:
    - in_channels (int): 輸入特徵圖的通道數
    - reduction_ratio (int): 通道注意力機制的縮減率
    """
    def __init__(self, in_channels, reduction_ratio=Config.reduction):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 計算中間層通道數
        self.inter_channels = max(8, in_channels // reduction_ratio)
        
        # 定義卷積層
        self.conv_h = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.conv_w = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # 定義BatchNorm
        self.bn_h = nn.BatchNorm2d(self.inter_channels)
        self.bn_w = nn.BatchNorm2d(self.inter_channels)
        
        # 定義最後的投影層
        self.conv_proj_h = nn.Conv2d(self.inter_channels, in_channels, 1)
        self.conv_proj_w = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向傳播函數
        
        Args:
            x (Tensor): 形狀為 (batch_size, channels, height, width) 的輸入特徵圖
            
        Returns:
            Tensor: 經過注意力增強的特徵圖
        """
        batch_size, channels, height, width = x.size()
        
        # 計算水平方向的注意力
        # 首先在垂直方向上進行平均池化
        x_h = F.adaptive_avg_pool2d(x, (height, 1))
        x_h = self.conv_h(x_h)
        x_h = self.bn_h(x_h)
        
        # 計算垂直方向的注意力
        # 首先在水平方向上進行平均池化
        x_w = F.adaptive_avg_pool2d(x, (1, width))
        x_w = self.conv_w(x_w)
        x_w = self.bn_w(x_w)
        
        # 轉置操作以對齊維度
        x_w = x_w.transpose(2, 3)
        
        # 生成注意力圖
        h_att = self.conv_proj_h(x_h).sigmoid()
        w_att = self.conv_proj_w(x_w).transpose(2, 3).sigmoid()
        
        # 組合兩個方向的注意力
        attention = h_att * w_att
        
        # 應用注意力到輸入特徵
        out = x * attention
        
        return out

#通道空間注意力機制CBAM: Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=Config.reduction_1):
        """
        通道注意力模塊初始化
        Args:
            in_channels: 輸入特徵圖的通道數
            reduction_ratio: 降維比例，用於減少計算量
        """
        super(ChannelAttention, self).__init__()
        
        # 定義共享MLP網絡
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        """
        前向傳播過程
        Args:
            x: 輸入特徵圖 [batch_size, channels, height, width]
        """
        # 全局平均池化
        avg_pool = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        # 全局最大池化
        max_pool = F.max_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        
        # 分別通過共享MLP
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        
        # 將兩個特徵相加後通過sigmoid激活函數
        out = torch.sigmoid(avg_out + max_out)
        
        # 調整維度以便於後續運算
        return out.view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        空間注意力模塊初始化
        Args:
            kernel_size: 卷積核大小，論文中使用7x7
        """
        super(SpatialAttention, self).__init__()
        
        # 確保kernel_size為奇數
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        padding = kernel_size // 2
        
        # 定義7x7卷積層
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        
    def forward(self, x):
        """
        前向傳播過程
        Args:
            x: 輸入特徵圖 [batch_size, channels, height, width]
        """
        # 計算平均值特徵圖
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 計算最大值特徵圖
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接特徵圖
        x = torch.cat([avg_out, max_out], dim=1)
        
        # 通過卷積層和sigmoid激活函數
        x = torch.sigmoid(self.conv(x))
        return x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=Config.reduction_1, kernel_size=7):
        """
        CBAM模塊初始化
        Args:
            in_channels: 輸入特徵圖的通道數
            reduction_ratio: 通道注意力模塊中的降維比例
            kernel_size: 空間注意力模塊中的卷積核大小
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        """
        前向傳播過程
        Args:
            x: 輸入特徵圖
        """
        # 先進行通道注意力
        x = x * self.channel_attention(x)
        # 再進行空間注意力
        x = x * self.spatial_attention(x)
        return x

class CustomFeatureFusion(nn.Module):
    def __init__(self):
        super(CustomFeatureFusion, self).__init__()
        
    def kurtosis_index(self, signal):
        # 直接使用 PyTorch 操作
        mean_val = torch.mean(signal)
        variance = torch.mean((signal - mean_val)**2)
        return torch.mean((signal - mean_val)**4) / (variance**2)
    
    def cross_correlation(self, x_i, x_j):
        # 使用 PyTorch 計算交叉相關
        N = x_i.numel()
        return torch.sum(x_i * x_j) / N
    
    def cross_correlation_energy(self, x_signals):
        k = len(x_signals)
        E_matrix = torch.zeros((k, k), device=x_signals[0].device)
        E_total = torch.zeros(k, device=x_signals[0].device)
        
        for i in range(k):
            for j in range(k):
                if i != j:
                    E_matrix[i, j] = self.cross_correlation(
                        x_signals[i].flatten(), 
                        x_signals[j].flatten()
                    )**2
            E_total[i] = torch.sqrt(torch.sum(E_matrix[i]))
        return E_matrix, E_total
    
    def weighted_fusion(self, x_signals, E_total, kurtosis_indices):
        # 使用 PyTorch 操作進行加權融合
        weights = E_total * kurtosis_indices
        weights = weights / torch.sum(weights)
        weights = weights.view(-1, 1, 1, 1)
        
        # 加權求和
        fused_signal = sum([weights[i] * x_signals[i] for i in range(len(x_signals))])
        return fused_signal
    
    def forward(self, x1, x2):
        x_signals = [x1, x2]
        
        # 計算峭度指數
        kurtosis_indices = torch.tensor([
            self.kurtosis_index(x1),
            self.kurtosis_index(x2)
        ], device=x1.device)
        
        # 計算交叉相關能量
        _, E_total = self.cross_correlation_energy(x_signals)
        # 計算交叉相關能量
        #E_matrix, E_total = self.cross_correlation_energy(x_signals)    
        # 執行加權融合
        fused_features = self.weighted_fusion(x_signals, E_total, kurtosis_indices)
        
        return fused_features

# 定義 MobileNetV3 Block
# 定義 Hard-Swish 激活函數
class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

# 定義 MobileNetV3 Block
class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, use_se=False):
        super(MobileNetV3Block, self).__init__()
        self.use_se = use_se  # 是否使用 Squeeze-and-Excitation 模塊

        # 1x1 Pointwise Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Depthwise Convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                               padding=kernel_size//2, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 Pointwise Convolution
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 激活函數 (ReLU 或 Hard-Swish)
        self.activation = activation()

        # Squeeze-and-Excitation 模塊
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        identity = x  # 儲存輸入值，以進行殘差連接
    
        # Pointwise Convolution (1x1)
        x = self.activation(self.bn1(self.conv1(x)))
        # Depthwise Convolution
        x = self.activation(self.bn2(self.conv2(x)))
    
        # Squeeze-and-Excitation 模塊（若使用）
        if self.use_se:
            x = x * self.se(x)
    
        # Pointwise Convolution (1x1)
        x = self.bn3(self.conv3(x))
    
        # 加入殘差連接（輸入輸出尺寸必須相同：通道數相同且stride=1）
        if identity.shape == x.shape:
            x += identity

        return x


# 定義 MobileNetV3 主架構
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(MobileNetV3, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs = HardSwish()  # 使用 Hard-Swish 激活函數

        # 定義 MobileNetV3 Blocks
        self.blocks = nn.Sequential(
            MobileNetV3Block(16, 16, kernel_size=3, stride=1, activation=nn.ReLU, use_se=False),
            MobileNetV3Block(16, 24, kernel_size=3, stride=2, activation=nn.ReLU, use_se=False),
            MobileNetV3Block(24, 24, kernel_size=3, stride=1, activation=nn.ReLU, use_se=False),
            MobileNetV3Block(24, 40, kernel_size=5, stride=2, activation=HardSwish, use_se=True),
            MobileNetV3Block(40, 40, kernel_size=5, stride=1, activation=HardSwish, use_se=True),
            MobileNetV3Block(40, 80, kernel_size=3, stride=2, activation=HardSwish, use_se=False),
            MobileNetV3Block(80, 112, kernel_size=3, stride=1, activation=HardSwish, use_se=True),
            MobileNetV3Block(112, 160, kernel_size=5, stride=2, activation=HardSwish, use_se=True)
        )
        # 保留圖像增強和特徵融合組件
        self.image_enhancement = AdaptiveImageEnhancement()
        # 特徵融合模組
        self.fusion = CustomFeatureFusion()
        #通道空間注意力機制
        self.attention_1 = CBAM(160)
        # 加入座標注意力機制
        self.attention = CoordinateAttention2D(160)

        # 最後的 1x1 卷積 + 池化
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 添加 Batch Normalization 層
        self.bn_final = nn.BatchNorm1d(960)
        self.dropout = nn.Dropout(0.2)  # 添加 dropout

        self.fc = nn.Linear(960, num_classes)
        

    def forward(self, x1):
        # 路徑2增強
        x11 = self.image_enhancement(x1)    # 路徑2增強
        x = self.fusion(x1, x11)    # 特徵融合 = 路徑1 + 路徑2
        x = self.hs(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.hs(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.shape[0], -1)
        x = self.bn_final(x)  # 添加最後的 BN
        x = self.dropout(x)   # 添加 dropout
        x = self.fc(x)
        #x = self.attention_1(x)# CBAM注意力機制
        return x


# 主程式
if __name__ == "__main__":
    # === 1. 首先添加 GPU 診斷代碼 ===
    print("="*50)
    print("GPU 診斷信息：")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 設備數量: {torch.cuda.device_count()}")
    print(f"設備名稱: {torch.cuda.get_device_name(0)}")
    print("="*50)
    # 確認GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 設置隨機種子確保可重現性
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # 資料集準備
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    transform = transforms.Compose([transforms.Resize(Config.IMAGE_SIZE), transforms.ToTensor()])
    full_dataset = PCBDataset(Config.IMAGE_DIR, Config.LABEL_DIR, transform=transform)
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]


    # 交叉驗證設定
    #best_test_loss = float('inf') # 追踪最佳驗證損失
    best_test_acc = 0  # 追踪最佳驗證準確度
    best_model_state = None  # 追踪最佳模型的狀態

    all_f1_scores, all_precisions, all_recalls = [], [], []
    all_train_losses, all_test_losses = [], []
    all_train_accs, all_test_accs = [], []
    all_f1_scores, all_precisions, all_recalls = [], [], []
    # 初始化標記，檢查是否為第一個fold
    is_first_fold = True
    is_first_fold1 = True
    # K折交叉驗證
    skf = StratifiedKFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold + 1}/{Config.K_FOLDS}")
        # 準備數據加載器
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True)

        # 初始化模型和優化器
        model = MobileNetV3(num_classes=Config.NUM_CLASSES).to(device)
        #criterion = nn.CrossEntropyLoss().to(device)
        #optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, eps=1e-7, amsgrad=True)
        
        # ---- 使用 Adam + label smoothing crossentropy ----2025/3/13
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        #optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, eps=1e-7, amsgrad=True, weight_decay=1e-2)
        optimizer = optim.RMSprop(model.parameters(), lr=Config.LEARNING_RATE)

        # 初始化 GradScaler
        scaler = GradScaler('cuda')
       
        # 訓練循環
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Train]") as t:
                for images, labels in t:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    # 添加梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    
                    t.set_postfix({
                        'Loss': f"{train_loss/(t.n+1):.4f}",
                        'Acc': f"{100.*train_correct/train_total:.2f}%"
                    })

                # 計算並保存訓練指標
                epoch_train_loss = train_loss / len(train_loader)
                epoch_train_acc = 100. * train_correct / train_total
                train_losses.append(epoch_train_loss)
                train_accuracies.append(epoch_train_acc)

                # 驗證階段
                model.eval()
                test_loss, test_correct, test_total = 0, 0, 0
                y_true, y_pred = [], []
                
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()
                        
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())

                # 計算並保存驗證指標
                epoch_test_loss = test_loss / len(test_loader)
                epoch_test_acc = 100. * test_correct / test_total
                test_losses.append(epoch_test_loss)
                test_accuracies.append(epoch_test_acc)

                #print(f'\nEpoch {epoch+1}/{Config.EPOCHS}:')
                print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
                print(f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%\n')

                # ---- 更新 LR Scheduler ----2025/3/13
                #scheduler.step()
        # 保存損失與準確率摺線圖（確保有數據後才繪圖）
        if len(train_losses) > 0 and len(test_losses) > 0:
            plt.figure(figsize=(10, 8))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            loss_plot_path = os.path.join(Config.OUTPUT_DIR, f'fold_{fold + 1}_loss_curve.jpg')
            plt.savefig(loss_plot_path, dpi=1200)
            plt.close()
    
        if len(train_accuracies) > 0 and len(test_accuracies) > 0:
            plt.figure(figsize=(10, 8))
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
            plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid()
            accuracy_plot_path = os.path.join(Config.OUTPUT_DIR, f'fold_{fold + 1}_accuracy_curve.jpg')
            plt.savefig(accuracy_plot_path, dpi=1200)
            plt.close()
    
            # 繪製混淆矩陣並保存
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cm_path = os.path.join(Config.OUTPUT_DIR, f'fold_{fold + 1}_confusion_matrix_1.jpg')
            plt.savefig(cm_path, dpi=1200)
            plt.close()
            
            # 計算混淆矩陣 百分比圖
            cm = confusion_matrix(y_true, y_pred)
            # 計算百分比
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            # 將數值轉換為帶有 "%" 的格式
            annot_labels = np.array([["{:.2f}%".format(value) for value in row] for row in cm_percentage])
            # 繪製熱圖
            plt.figure(figsize=(10,8))
            sns.heatmap(cm_percentage, annot=annot_labels, fmt="", cmap="Blues", cbar=True, xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cm_path = os.path.join(Config.OUTPUT_DIR, f'fold_{fold + 1}_confusion_matrix.jpg')
            plt.savefig(cm_path, dpi=1200)
            plt.close()
    
            # 保存模型
            model_save_path = os.path.join(Config.OUTPUT_DIR, f'fold_{fold + 1}_model.h5')
            torch.save(model.state_dict(), model_save_path)
        # 保存指標到 CSV 檔案
        metrics_csv_path = os.path.join(Config.Accuracy, 'metrics_0.01.csv')
        # 開啟 CSV 檔案進行寫入，使用 'a' 模式來避免覆蓋
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # 如果是第一次寫入（即五折完成後），寫入設定值和標題行
            if is_first_fold:
                writer.writerow([])  # 空行分隔設定值和實際的結果
                writer.writerow(['EPOCHS',Config.EPOCHS,'BATCH_SIZE',Config.BATCH_SIZE,'LEARNING_RATE',Config.LEARNING_RATE])
                writer.writerow(['Fold', 'F1 Score', 'Precision', 'Recall', 'Train Loss', 'Train Accuracy', 'Testing Loss', 'Testing Accuracy'])
                is_first_fold = False  # 標記為非第一次寫入          
            # 計算指標
            classification_rep = classification_report(y_true, y_pred, target_names=Config.CLASS_NAMES, zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted')
            # 寫入每一折的評估指標
            writer.writerow([fold + 1, f1, precision, recall, train_losses[-1], train_accuracies[-1], test_losses[-1], test_accuracies[-1]])
            # 儲存每一折的結果到列表
            all_f1_scores.append(f1)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_train_losses.append(train_losses[-1])
            all_test_losses.append(test_losses[-1])
            all_train_accs.append(train_accuracies[-1])
            all_test_accs.append(test_accuracies[-1])

    # 保存總平均結果到 CSV（這部分可以寫入不同檔案）
    final_metrics_csv_path = os.path.join(Config.Accuracy, 'final_metrics_0.01.csv')
    with open(final_metrics_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)    
        if is_first_fold1:
            writer.writerow([])  # 空行分隔設定值和實際的結果
            writer.writerow(['EPOCHS', Config.EPOCHS, 'BATCH_SIZE', Config.BATCH_SIZE, 'LEARNING_RATE', Config.LEARNING_RATE])
            writer.writerow(['Metric', 'Average', 'Std Dev'])
            is_first_fold1 = False           
        # 寫入各項平均值和標準差
        writer.writerow(['Training Loss', np.mean(all_train_losses), np.std(all_train_losses)])
        writer.writerow(['Testing Loss', np.mean(all_test_losses), np.std(all_test_losses)])
        writer.writerow(['Training Accuracy', np.mean(all_train_accs), np.std(all_train_accs)])
        writer.writerow(['Testing Accuracy', np.mean(all_test_accs), np.std(all_test_accs)])
        writer.writerow(['F1 Score', np.mean(all_f1_scores), np.std(all_f1_scores)])
        writer.writerow(['Precision', np.mean(all_precisions), np.std(all_precisions)])
        writer.writerow(['Recall', np.mean(all_recalls), np.std(all_recalls)])
        
# === 完成本組參數交叉驗證後，釋放該組參數相關佔用的記憶體 ===
# 此處注意：因為 model、optimizer、DataLoader 都是在各個 fold 中被建立，
# 最後一次建立的物件我們在這裡刪除就足夠（其他 fold 的局部變數已隨迴圈結束自動超出作用域）
del model, optimizer, train_loader, test_loader, full_dataset
gc.collect()
torch.cuda.empty_cache()
print("所有實驗已完成，將重啟 Python 內核以釋放記憶體...")
