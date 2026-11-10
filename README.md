# Gaia 星场模拟观测数据生成

本仓库保存 `Dancecam/data/generated/` 中两个数据集所用的生成代码、配置和星团清单。输出是**模拟观测数据**，不是真实望远镜曝光，也不是 DanceCam 论文作者未公开的原始训练集。

## 两个生成任务

| 入口模块 | 星团清单 | 输出目录 | 额外输出 |
| --- | --- | --- | --- |
| `dancecam.opencluster_batch` | `Hunt_Reffert_2024_Gaia_100_candidates.md` | `data/generated/opencluster` | PNG 预览、`split_5000.json` |
| `dancecam.mutian_batch` | `recovered_mutian_193_r_100_candidates.md` | `data/generated/mutian_193_r` | 无 |

两者都使用 `configs/mutian_193_r.json`：从 100 个星团中心出发，每个中心取 50 个不同天空指向，计划各生成 5000 对 FITS/NPZ 文件。每条序列为 64 帧、每帧 256×256 像素。

代码分工：`cluster_video.py` 查询 Gaia DR3 并将恒星投影到探测器坐标；`mutian_video.py` 模拟大气湍流、光学成像、曝光积分和噪声；`simulate.py` 写入 FITS；`fits_to_npz.py` 生成压缩的训练副本。Gaia G 星等被用作 R 波段亮度的近似值，具体望远镜和大气参数见 JSON 配置。

## 安装与运行

使用 Python 3.11 或 3.12、支持 CUDA 的 PyTorch，以及有足够显存的 GPU。在仓库根目录运行：

```powershell
python -m pip install -e .
python -u -m dancecam.opencluster_batch --config configs/mutian_193_r.json --catalog data/source/catalogs/Hunt_Reffert_2024_Gaia_100_candidates.md --out data/generated/opencluster --device cuda
python -u -m dancecam.mutian_batch --config configs/mutian_193_r.json --catalog data/source/catalogs/recovered_mutian_193_r_100_candidates.md --out data/generated/mutian_193_r --device cuda
```

两个任务请分别运行。首次运行会访问 ARI Gaia TAP 服务，并把查询结果缓存在各自的输出目录。再次执行同一命令会检查已有 FITS 文件，再继续缺失的条目。续跑时不要修改配置、星团清单或 Gaia 缓存。脚本会在剩余磁盘空间低于 10 GiB 前停止；完整数据集还需要大量磁盘空间。这两个命令只生成数据，不启动模型训练。

每个 `<名称>.fits` 包含模拟视频、无噪声真值 `GROUNDTRUTH`、恒星表和完整配置；对应的 `sequence_<名称>.npz` 是供训练使用的无损压缩副本。`opencluster` 还生成 `<名称>.png` 预览。`manifest_5000.json` 记录已生成条目，`batch_status.json` 记录进度。`opencluster` 的划分文件将相互重叠的天空区域归入同一训练、验证或测试集合。

生成的数据、Gaia 查询缓存、日志和模型检查点均由 `.gitignore` 排除，不上传到 GitHub。若需逐字节重现输出，还需要保留相同的 Gaia 查询结果，并使用兼容的数值软件与硬件环境。
