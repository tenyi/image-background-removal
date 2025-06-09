import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import sys
import os

from torchvision.models.segmentation.deeplabv3 import DeepLabV3

def get_best_device() -> torch.device:
    """
    檢查系統支援的最佳運算裝置。
    
    優先順序：CUDA GPU > MPS (Apple Silicon) > ROCm (AMD GPU) > CPU
    
    回傳:
        torch.device: 最佳可用的運算裝置。
    """
    # 檢查 CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 使用 CUDA GPU: {torch.cuda.get_device_name()}")
        return device
    
    # 檢查 MPS (Apple Silicon GPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 使用 Apple Silicon MPS GPU")
        return device
    
    # 檢查 ROCm (AMD GPU) - 透過檢查是否有 ROCm 後端
    if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        try:
            device = torch.device("cuda")  # ROCm 使用 cuda 介面
            print("🚀 使用 AMD ROCm GPU")
            return device
        except:
            pass
    
    # 檢查其他可能的 GPU 後端
    if torch.cuda.is_available():  # 再次檢查以防萬一
        device = torch.device("cuda")
        print(f"🚀 使用 GPU: {torch.cuda.get_device_name()}")
        return device
    
    # 預設使用 CPU
    device = torch.device("cpu")
    print("💻 使用 CPU (未偵測到 GPU 支援)")
    return device

def use_gpu() -> bool:
    """
    檢查系統是否支援任何 GPU 加速。
    
    回傳:
        bool: 如果支援任何 GPU 則為 True，否則為 False。
    """
    # 檢查 CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return True
    
    # 檢查 MPS (Apple Silicon GPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return True
    
    # 檢查 ROCm (AMD GPU)
    if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return True
    
    return False

def preprocess_image(img_path: str):
    """
    讀取影像檔案並進行預處理，轉換為 PIL Image 和 PyTorch Tensor。

    參數:
        img_path (str): 輸入影像的路徑。

    回傳:
        tuple[Image.Image, torch.Tensor]: 包含原始 PIL Image 和預處理後的 PyTorch Tensor。
    """
    image = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(image).unsqueeze(0)  # 加 batch 維度
    return image, tensor

def run_deeplab(img_path: str, output_path: str, device: torch.device) -> None:
    """
    使用 DeepLabV3 模型進行影像背景移除。

    參數:
        img_path (str): 輸入影像的路徑。
        output_path (str): 輸出影像的儲存路徑。
        device (torch.device): 指定運行的設備 (CPU、CUDA GPU、MPS、ROCm 等)。
    """
    print("👉 使用 DeepLabV3 進行去背")

    model: DeepLabV3 = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT).to(device)
    model.eval()

    orig_image, input_tensor = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    # 將人物 class (VOC: class 15) 當前景
    mask = (output_predictions == 15).astype(np.uint8) * 255

    # 將 mask 應用到原圖
    orig_np = np.array(orig_image)
    alpha = mask.astype(np.uint8)
    result = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGRA)
    result[:, :, 3] = alpha

    cv2.imwrite(output_path, result)
    print(f"✅ 儲存去背結果於：{output_path}")

def run_grabcut_opencv(img_path: str, output_path: str):
    """
    使用 OpenCV 的 GrabCut 演算法進行影像背景移除。

    參數:
        img_path (str): 輸入影像的路徑。
        output_path (str): 輸出影像的儲存路徑。
    """
    print("👉 使用 OpenCV GrabCut 簡易去背")

    image = cv2.imread(img_path)
    mask = np.zeros(image.shape[:2], np.uint8)
    # 初始化背景/前景模型
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # 假設主體在中央，擷取中央 90% 區域
    height, width = image.shape[:2]
    rect = (int(width * 0.05), int(height * 0.05),
            int(width * 0.9), int(height * 0.9))

    # 使用 GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # mask 處理
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    output = image * mask2[:, :, np.newaxis]

    # 轉換為含 alpha 的格式
    alpha = (mask2 * 255).astype(np.uint8)
    result = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = alpha

    cv2.imwrite(output_path, result)
    print(f"✅ 儲存去背結果於：{output_path}")

def main() -> None:
    """
    主函式，根據最佳可用裝置來決定使用 DeepLabV3 或 GrabCut 進行影像背景移除。
    """
    if len(sys.argv) < 3:
        print("用法: python remove_bg.py input.jpg output.png")
        return

    img_path: str = sys.argv[1]
    output_path: str = sys.argv[2]

    if not os.path.exists(img_path):
        print("❌ 找不到輸入圖片。")
        return

    device = get_best_device()
    
    if device.type != "cpu":
        run_deeplab(img_path, output_path, device)
    else:
        print("⚠️  未偵測到 GPU 支援，使用 CPU 版本的 GrabCut 演算法")
        run_grabcut_opencv(img_path, output_path)

if __name__ == "__main__":
    main()
