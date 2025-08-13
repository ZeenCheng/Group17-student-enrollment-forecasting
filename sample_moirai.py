import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

MODEL = "moirai-moe"  # 'moirai' or 'moirai-moe'
SIZE = "small"
PDT = 20        # 预测长度
CTX_BASE = 100  # 最小上下文长度起点，可以调节
CTX_MAX = 200   # 最大上下文长度限制，防止超长
PSZ = "auto"
BSZ = 32
TEST = 100

url = "ts_wide.csv"
df = pd.read_csv(url, index_col=0, parse_dates=True)

ds = PandasDataset(dict(df))
train, test_template = split(ds, offset=-TEST)

# 加载模型（预训练）
if MODEL == "moirai":
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX_MAX,  # 设置最大context_length
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
else:
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX_MAX,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

predictor = model.create_predictor(batch_size=BSZ)

# 总测试窗口数
windows = TEST // PDT

all_forecasts = []
all_labels = []
all_inputs = []

for w in range(windows):
    # 动态上下文长度，逐渐扩大，最大不超过CTX_MAX
    current_ctx = min(CTX_BASE + w * PDT, CTX_MAX)

    # 计算当前窗口的起始offset
    # 这里的offset代表切分点，确保训练集包含足够上下文长度
    offset = -(TEST - w * PDT)

    # 重新切分训练集和测试集
    train_w, test_w = split(ds, offset=offset)

    # 生成测试实例，windows=1 只取当前窗口一个样本
    instances = test_w.generate_instances(
        prediction_length=PDT,
        windows=1,
        distance=1,
        context_length=current_ctx  # 动态传入上下文长度（很重要）
    )
    instance_list = list(instances)
    if len(instance_list) == 0:
        print(f"窗口 {w} 无测试样本，跳过")
        continue

    # 取第一个实例
    test_instance = instance_list[0]
    # test_instance是一个命名元组，包含input和label两个部分
    # 注意：此处的结构是 (input_dict, label_tensor)
    input_data, label = test_instance

    # 预测（predictor支持传入字典格式输入）
    forecast = next(predictor.predict([input_data]))

    all_forecasts.append(forecast)
    all_labels.append(label)
    all_inputs.append(input_data)

# 画第一个窗口预测示例
plot_single(
    all_inputs[0],
    all_labels[0],
    all_forecasts[0],
    context_length=min(CTX_BASE, CTX_MAX),
    name="pred_expanding_window_dynamic_ctx",
    show_label=True,
)
plt.show()
