# Push-T Expert Planner

**Contact-Aware Hybrid Expert Planning for Robotic T-Shaped Object Pushing**

在 MuJoCo / robosuite 中控制 Franka Panda 机械臂推动 T 形物体，生成用于模仿学习的专家状态–动作轨迹。初始角度误差覆盖正负 0–180°，支持多进程采集、仿真状态缓存和代表性轨迹录像。

## 1. 算法

混合专家控制器结合分阶段基线控制与接触感知多步联合规划，在有限执行时间内寻找满足任务成功条件的物理轨迹。

1. **分阶段基线**：使用标定接触响应和闭环旋转、平移控制完成任务，同时记录真实中间状态。
2. **联合位姿规划**：基于 T 块几何、接触点、可达范围和剩余时间，预测多个推动动作对位置与角度的联合影响。
3. **物理反馈搜索**：当基线未成功时，对候选动作执行物理仿真，测量实际位姿变化、修正接触响应并重新规划。可选的针对性恢复搜索根据实际终点拟合推动距离、尝试分段旋转并剪除饱和动作。
4. **状态缓存**：保存仿真与控制器状态，复用搜索分支的历史轨迹，定期核对恢复一致性；异常时使用完整重放。
5. **轨迹验证**：只导出满足成功与稳定性要求的轨迹。搜索成功轨迹须经过完整动作重放验证。

任务上限为 400 个控制步（20 Hz），每个动作 7 维。环境成功条件为 IoU ≥ 0.85、物体直立且在桌面上；专家数据默认还要求连续 10 步成功且末端达到安全抬升高度。分阶段基线内部采用有限接触序列，混合专家不限制总推动次数。离线搜索的额外仿真步数不计入最终专家轨迹的 400 步预算。

## 2. 环境准备

建议沿用已能运行 Push-T 的 Python 环境。依赖见 `requirements.txt`。两份接触标定文件位于 `config/`，无需重新标定。

```bash
python check_setup.py
```

## 3. 生成专家数据

先测试 24 个随机场景，并只录制 4 个代表性视频：

```bash
python collect_expert.py --fresh --wide-angles --per-bin 4 --video --video-count 4
```

正式采集 0–90°（600 个初始场景）：

```bash
python collect_expert.py --collect --per-bin 100 --workers 4 --trials 40 --rescue-trials 0 --cache-nodes 16 --no-video
```

正式采集 90–180°（600 个初始场景）：

```bash
python collect_expert.py --collect --wide-angles --per-bin 100 --workers 4 --trials 40 --rescue-trials 0 --cache-nodes 16 --no-video
```

每 15° 为一个区间，正负角度均衡；实际成功轨迹数量取决于物理仿真结果。正式采集配置默认建议关闭额外恢复搜索，优先提高有效数据吞吐量。若需要针对困难场景进行补充，可设置 `--rescue-trials 24`。

| 参数 | 用途 |
|---|---|
| `--fresh` / `--collect` | 小规模试运行 / 正式采集；未指定模式时默认小规模运行 |
| `--wide-angles` | 采样绝对角度误差 90–180°，否则为 0–90° |
| `--per-bin` | 每个角度区间的场景数，必须为正偶数 |
| `--seed` | 起始环境 seed |
| `--workers` | 独立仿真进程数 |
| `--trials` | 每个失败场景的常规搜索预算 |
| `--rescue-trials` | 额外恢复搜索预算，0 表示关闭 |
| `--cache-nodes` | 每个仿真进程的缓存节点数，0 表示关闭 |
| `--cases-file` | 读取已有 `cases.csv`，复现完全相同的初始场景 |
| `--output` | 自定义输出目录 |
| `--video --video-count N` | 预选 N 个典型案例录像；录像使用单进程 |
| `--plan-only` | 仅生成场景清单，不运行仿真 |

其他运行参数可通过 `python collect_expert.py --help` 查看。录像不影响专家数据筛选。默认结果路径不会覆盖已存在的批次结果。

## 4. 项目结构

```text
push_t_expert/
├── README.md
├── requirements.txt
├── collect_expert.py              # 唯一任务执行与数据采集入口
├── check_setup.py                 # 环境及标定文件检查
├── config/
│   ├── contact_model.json         # 旋转接触标定
│   └── contact_response_model.json # 平移与转角联合响应标定
└── src/
    ├── environment.py             # Push-T 环境及随机初始化配置
    ├── geometry.py                # T 块几何、接触候选与 IoU
    ├── rotation_controller.py     # 旋转与机器人运动控制基元
    ├── translation_controller.py  # 闭环平移、接触保持与动态修正
    ├── contact_planner.py         # 标定模型及多接触序列规划
    ├── baseline.py                # 分阶段控制与旋转/平移切换
    ├── joint_lookahead.py         # 多步联合位姿预测
    ├── expert_search.py           # 混合搜索与批量采集调度
    ├── physical_feedback.py       # 实测响应及针对性恢复规划
    ├── sim_checkpoint.py          # 状态缓存、一致性核对与回退
    └── dataset.py                 # 场景采样、轨迹导出与录像
```

## 5. 数据输出

结果默认保存于 `results/fresh/` 或 `results/collect/`，大角度采集带 `_90_180` 后缀。指定场景清单时使用 `results/cases/`。

| 文件 | 内容 |
|---|---|
| `demos/*.npz` | 通过验证的成功专家轨迹 |
| `summary.csv` | 每个场景的成功率指标、IoU、步数、耗时与缓存统计 |
| `cases.csv` / `run_config.json` | 初始场景与运行配置 |
| `search_trials.csv` | 物理搜索分支与实际接触反馈 |
| `traces/` / `case_reports/` | 轨迹诊断与单场景报告 |
| `videos/` | 可选 MP4 录像 |

每条专家轨迹的长度 T ≤ 400，包含：

| 数组 | 含义 |
|---|---|
| `actions` | T × 7 控制动作 |
| `obs/<key>` | T+1 帧原环境数值观测 |
| `stage`, `phase` | T 个控制阶段和动作阶段标签 |
| `success`, `iou` | T+1 帧任务状态 |
| `task_target_xy_yaw` | T+1 帧目标位姿 |
| `metadata_json` | seed、初始角度、算法与标定摘要等元数据 |

```python
import numpy as np

with np.load("results/collect/demos/example.npz", allow_pickle=False) as demo:
    actions = demo["actions"]
    observations = {key[4:]: demo[key] for key in demo.files if key.startswith("obs/")}
```

第 t 个动作对应第 t 帧观测，下一帧为执行后的观测。当前导出为状态型模仿学习数据，MP4 录像不是同步视觉训练数据。建议按完整轨迹与 seed 划分训练、验证和测试集，并检查各角度区间的覆盖率。

