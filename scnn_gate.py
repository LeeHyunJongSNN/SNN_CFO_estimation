import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron  # spikingjelly 라이브러리 사용

# Spikingjelly를 활용한 Spiking Linear Layer (IF 뉴런 기반)
class SpikingLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, threshold=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.if_node = neuron.IFNode(v_threshold=threshold)  # IF 뉴런

    def forward(self, x):
        x = self.linear(x)
        x = self.if_node(x)
        return x

# Hard Gate: 두 분기 중 하나만 선택 (Gumbel Softmax with hard=True 사용)
class HardGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_params = nn.Parameter(torch.randn(2))

    def forward(self, branch1, branch2):
        # Gumbel softmax를 통해 hard selection 수행 (하나는 1, 하나는 0인 one-hot vector 반환)
        gate = F.gumbel_softmax(self.gate_params, tau=1.0, hard=True)
        # one-hot vector를 사용해 한 분기의 출력만 선택
        return gate[0] * branch1 + gate[1] * branch2

# SNN 기반 CFO 추정 모델 (논문 구조 기반)
class SNN_CFO_Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Convolution Part ---
        # 입력 벡터 (320)를 (batch, 1, 10, 32)로 reshape하여 2D convolution 적용
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # 2x2 max pooling
        self.relu = nn.ReLU()
        
        # --- Linear Part: 첫 번째 블록 (640 -> 512) ---
        self.fc1_perc = nn.Linear(640, 512)             # DNN (퍼셉트론) 분기
        self.fc1_spike = SpikingLinear(640, 512, threshold=1.0)  # SNN 분기 (spikingjelly 사용)
        self.gate1 = HardGate()  # 두 분기 중 하나만 선택
        
        # --- Linear Part: 두 번째 블록 (512 -> 128) ---
        self.fc2_perc = nn.Linear(512, 128)
        self.fc2_spike = SpikingLinear(512, 128, threshold=1.0)
        self.gate2 = HardGate()
        
        # --- Prediction Part: 최종 예측 (128 -> 1) ---
        self.fc_pred = nn.Linear(128, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        # 입력 벡터 (320)를 (batch, 1, 10, 32)로 reshape
        x = x.view(batch_size, 1, 10, 32)
        x = self.conv(x)         # (batch, 8, 10, 32)
        x = self.pool(x)         # (batch, 8, 5, 16) → 8*5*16 = 640
        x = self.relu(x)
        x = x.view(batch_size, -1)  # flatten → (batch, 640)
        
        # --- 첫 번째 선형 블록 (게이트로 분기 선택) ---
        branch1 = self.fc1_perc(x)     # DNN 분기
        branch2 = self.fc1_spike(x)      # SNN 분기 (spikingjelly 사용)
        x = self.gate1(branch1, branch2)  # 두 분기 중 하나 선택
        x = self.relu(x)
        
        # --- 두 번째 선형 블록 (게이트로 분기 선택) ---
        branch1 = self.fc2_perc(x)
        branch2 = self.fc2_spike(x)
        x = self.gate2(branch1, branch2)
        x = self.relu(x)
        
        # --- 예측 파트 ---
        out = self.fc_pred(x)  # 최종 CFO 추정 값 (실수 하나)
        return out

# 모델 예시 및 더미 데이터에 대한 전방 전달 예제
if __name__ == "__main__":
    model = SNN_CFO_Estimator()
    # (배치 사이즈, 320) 크기의 더미 입력 생성 (예: 배치 사이즈 100)
    dummy_input = torch.randn(100, 320)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (100, 1)
    print("Output:", output)
