import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def load_data(file_path):
    """AG數據讀取"""
    AG_data = []
    with open(f'{file_path}.txt', 'r') as file:
        for line in file:
            if line.startswith("mcu:"):
                values = line.strip().split(",")[1:]
                AG_data.append([float(v) for v in values])

    """IQ數據讀取及時間標準化(以第一筆數據為0)"""
    IQ_data = pd.read_csv(f'{file_path}.csv')
    time_parts = IQ_data['Time'].str.split('-', expand=True).astype(float)
    IQ_data['time'] = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
    IQ_data['time'] -= IQ_data['time'].iloc[0]
   
    return np.array(AG_data), IQ_data[['time', 'I', 'Q']]
    

def reshape_data(data, time_step, retain_substeps=None):
    """
    依照時間步長對數據進行分割和重塑

    參數:
    - data: ndarray, 包含要進行重塑的數據。
    - time_step: int, 每個分割塊的時間步長大小。
    - retain_substeps: int, 選擇保留每個時間塊中的部分步長(默認為 None, 即保留全部)。

    返回:
    - reshaped_data: ndarray, 依照時間步長分割並重塑的數據。
    """
    num_rows = data.shape[0]
    # 修剪數據，使其能夠整除 time_step
    data = data[:num_rows - (num_rows % time_step), :]
    reshaped_data = data.reshape(-1, time_step, data.shape[1])
    
    # 保留指定的步長，或者返回完整的重塑數據
    return reshaped_data[:, :retain_substeps, :] if retain_substeps else reshaped_data

def interpolate_data(sensor_data, target_length):
    """
    對加速度和角速度數據進行插值處理，以得到新的數據長度。

    參數:
    - sensor_data: ndarray, 包含加速度和角速度數據, 每列代表一個參數(6個參數)。
    - target_length: int, 插值後的目標數據長度。

    返回:
    - interpolated_data: ndarray, 插值後的數據, 具有指定的目標長度。
    """
    original_length = sensor_data.shape[0]
    original_index = np.arange(original_length)  # 使用原始索引作為插值基準
    new_index = np.linspace(0, original_length - 1, target_length)  # 生成插值後的新索引

    # 初始化插值后的數據
    interpolated_data = np.zeros((target_length, sensor_data.shape[1]))

    # 對每個參數（每列）進行插值
    for i in range(sensor_data.shape[1]):
        interp_func = interp1d(original_index, sensor_data[:, i], kind='cubic')  # 默認使用三次插值
        interpolated_data[:, i] = interp_func(new_index)

    return interpolated_data



# -------------------------------
# 假設以下函式已在其他地方定義好：
# 1) load_data(folder_path)     -> 回傳 (ag_data, iq_data)
# 2) interpolate_data(data, n)  -> 對 data 做插值到 n 點
# 3) reshape_data(data, N)      -> 將 data 切成 shape = (?, N, 6) 等等
# -------------------------------


# -------------------------------
# 產生 & 儲存單一感測器梅爾頻譜
# -------------------------------
def save_single_mel(sensor_data, sr, out_path):
    """
    給定單個感測器的一維訊號 sensor_data，
    計算並輸出梅爾頻譜圖 (不顯示，不包含任何標籤) 至 out_path。
    """
    # 計算梅爾頻譜
    mel_spectrogram = librosa.feature.melspectrogram(
        y=sensor_data.astype(float),
        sr=sr,
        n_fft=32,      # Fourier 變換大小，可依需求調整 要比N小
        hop_length=1,  # 跳躍點數，可依需求調整
        n_mels=12      # 梅爾濾波器數量，可依需求調整 要比n_fft 1/2-4
    )
    # 轉換到 dB
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 繪圖 (單張圖)
    fig, ax = plt.subplots(figsize=(4, 3))
    # 繪製梅爾頻譜，不顯示軸標籤或 colorbar 以符合「不包含標籤」的需求
    librosa.display.specshow(mel_spectrogram_db, sr=sr, ax=ax)

    ax.set_axis_off()  # 隱藏座標軸

    # 儲存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)  # 去除邊框
    plt.close(fig)  # 關閉圖表，釋放記憶體


# -------------------------------
# 將多筆樣本 (shape=(X, N, 6)) 輸出成多張圖
# -------------------------------
def export_data_to_images(data_array, class_name, out_dir, fs=20):
    """
    data_array: shape = (樣本數, N, 6)，
                每個樣本含 N 點、6 維度(6個感測器數據)
    class_name: 'class1' 或 'class2' 等
    out_dir:    基本輸出資料夾，例如 '/home/st424/CvT/DATASET/agmel/train' 
    fs:         採樣率(每秒多少點)，用於梅爾頻譜計算
    """
    for i, sample in enumerate(data_array):
        # sample shape = (N, 6)
        for d in range(6):
            sensor_data = sample[:, d]  # 第 d 個感測器訊號, shape=(N,)

            # 輸出路徑:  /home/st424/CvT/DATASET/agmel/train/classX/dimY/i.jpg
            out_path = os.path.join(out_dir, class_name, f"dim{d+1}", f"{i}.jpg")
            save_single_mel(sensor_data, sr=fs, out_path=out_path)


# -------------------------------
# 主流程：讀取「彎舉」與「側平舉」資料，做 train/val split，輸出圖像
# -------------------------------
if __name__ == "__main__":
    N = 40
    print('-- 彎舉數據 (Bicep Curl) --')
    Bicep_Curl_data2 = np.empty((0, N, 6))
    for x in range(1, 5):
        ag_data, iq_data = load_data(f'Dumbbell Bicep Curl/{x}')
        ag_data = interpolate_data(ag_data, ag_data.shape[0]*N//4)
        ag_data = reshape_data(ag_data, N)
        Bicep_Curl_data2 = np.vstack((Bicep_Curl_data2, ag_data))
    print('Bicep_Curl_data2 shape:', Bicep_Curl_data2.shape)

    print('-- 側平舉數據 (Lateral Raise) --')
    Lateral_Raise_data2 = np.empty((0, N, 6))
    for x, offset in enumerate([0, 0, 8, 6, 7, 7, 0], start=1):
        ag_data, iq_data = load_data(f'Dumbbell Lateral Raise/{x}')
        ag_data = ag_data[offset:, :]
        ag_data = interpolate_data(ag_data, ag_data.shape[0]*2)
        # reshape_data 的參數您可依實際需求調整
        ag_data = reshape_data(ag_data, N if x < 3 else 20*(N//4), None if x < 3 else N)
        Lateral_Raise_data2 = np.vstack((Lateral_Raise_data2, ag_data))
    print('Lateral_Raise_data2 shape:', Lateral_Raise_data2.shape)

    # 假設要輸出到以下根目錄 (請依實際需求調整)
    base_output_dir = "/home/st424/CvT/DATASET/agmel"

    # 這裡僅做示範：簡易 8:2 split
    # 您也可依自己需求手動分配哪幾筆屬於 train/val
    def train_val_split(data, ratio=0.8):
        split_idx = int(len(data) * ratio)
        return data[:split_idx], data[split_idx:]

    # 1) Bicep_Curl => class1
    bicep_train, bicep_val = train_val_split(Bicep_Curl_data2, ratio=0.8)
    # 2) Lateral_Raise => class2
    lateral_train, lateral_val = train_val_split(Lateral_Raise_data2, ratio=0.8)

    # 輸出到 /train/class1/dimX/*.jpg
    export_data_to_images(bicep_train,   class_name='class1', out_dir=os.path.join(base_output_dir, 'train'))
    export_data_to_images(lateral_train, class_name='class2', out_dir=os.path.join(base_output_dir, 'train'))

    # 輸出到 /val/class1/dimX/*.jpg
    export_data_to_images(bicep_val,   class_name='class1', out_dir=os.path.join(base_output_dir, 'val'))
    export_data_to_images(lateral_val, class_name='class2', out_dir=os.path.join(base_output_dir, 'val'))

    print("Done! 所有感測器的梅爾頻譜圖片已輸出至", base_output_dir)