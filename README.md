# 设计文件自动匹配、自动命名与网页审核系统

本地单机版完整流程为：目录扫描与图片解码 → 按文件夹分类 → 为每个非 PSD 分类选择文件夹参考图并手绘 Mask → bbox/padding 裁剪与 Masked OpenCLIP，PSD 执行 Composite Preview、pHash 与整图 Embedding 决胜 → 按分类 Hungarian 全局一对一匹配 → 置信度诊断 → 聚合审核 → 生成 Rename Plan。

## 支持的实际目录结构

系统会递归寻找唯一的 `主图` 目录，因此同时支持简单扁平结构和当前实际交付结构：

```text
项目文件夹/
├── psd/
└── 项目名-效果图/
    ├── 主图/
    ├── 双人模特图/
    ├── 白底模特图/
    ├── 细节展示图/
    ├── 聚会场景/
    └── 观赛图/
```

选择最外层的“项目文件夹”即可。PSD 可以与效果图目录同级，不要求和主图处在同一层。

## 设计原则

- 图片用途只由一级文件夹决定，不用 AI 猜分类。
- 同一分类内，一张图片最多归属一个商品，一个商品最多获得一张自动匹配图片。
- Hungarian Assignment 使用 Dummy 槽允许拒绝低质量分配，不为了配满而强行归属。
- PSD 使用 `psd-tools` 读取 composite preview，先计算 pHash；当结构哈希并列时由整图 OpenCLIP Embedding 决胜。
- 普通附图使用“品类＋图片分类”手绘 Mask Profile。系统不生成预设遮罩；未完成全部非 PSD 分类的 Mask 前，禁止启动匹配。原图不修改，Mask、裁剪图和特征只作为可追溯派生资产。
- 聚合审核支持全选当前结果、通过已选和全部通过当前筛选。整组、单文件或批量确认后会立即刷新 Rename Plan，避免确认状态与改名计划不同步。
- 自动匹配只更新数据库归属并生成 Rename Plan，不移动、删除、覆盖或重命名源文件。Rename Plan 存在待确认、文件缺失或目标冲突时整批禁止执行。
- SQLite 保存原路径、相似度、Top 5 候选、行列 margin、分配排名、算法版本、运行记录和人工覆盖。
- `.design_matcher/` 缓存缩略图、2048px 预览和主图 embedding，不把大图永久留在内存。

## 环境与安装

推荐 Python 3.11–3.13。PyTorch/OpenCLIP 的 Python 版本支持以其官方安装包为准。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[clip]"
```

仅运行非 OpenCLIP 测试或先查看界面，可安装：

```powershell
pip install -e ".[dev]"
```

## 局域网启动（当前默认）

```powershell
.\start-lan.cmd
```

也可以直接运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8088
```

本机打开 `http://127.0.0.1:8088`，同一局域网内其他电脑打开
`http://192.168.0.31:8088`。运行期间不能关闭启动命令窗口。首次真正匹配时
OpenCLIP 可能需要下载模型权重。

如果电脑以后连接到其他 Wi-Fi，`192.168.0.31` 可能变化。运行 `ipconfig` 查看新的
IPv4 地址，并继续使用端口 `8088` 即可。

如果 Windows 防火墙以后被启用，需要以管理员身份为 TCP 8088 添加入站允许规则。

可配置项：

```powershell
$env:DESIGN_MATCHER_LOW_CONFIDENCE="0.80"
$env:DESIGN_MATCHER_MIN_SCORE="0.80"
$env:DESIGN_MATCHER_MIN_MARGIN="0.01"
$env:DESIGN_MATCHER_UNMATCHED_SCORE="0.50"
$env:DESIGN_MATCHER_CLIP_MODEL="ViT-B-32"
$env:DESIGN_MATCHER_CLIP_PRETRAINED="openai"
$env:DESIGN_MATCHER_BATCH_SIZE="16"
```

匹配处理页会实时显示模型加载、主图特征、附图/PSD 特征、全局分配和诊断保存进度，
同时显示当前实际使用 `CPU` 还是 `CUDA 显卡`。

## 先做只读匹配验证（推荐）

此命令输出每个文件的 Top1/Top2/Top3，不创建批次、不改名：

```powershell
python scripts/verify_match.py "D:\设计目录" --category "观赛图" --limit 30
```

确认视觉匹配质量后，再从网页创建批次并启动全局匹配。

## 测试

```powershell
pytest
```

测试使用轻量颜色特征替身，不下载 OpenCLIP 权重，覆盖扫描、Top5持久化、Hungarian、Dummy拒绝、置信状态、分类内唯一和人工结果保护。

真实数据的新旧算法对照可执行：

```powershell
python scripts/benchmark_masked.py --batch 2 --reference-batch 1 --output data/benchmark_masked_batch2.json
```

## 数据与安全

- 数据库默认位于 `data/matcher.db`。
- 每个设计根目录下会生成 `.design_matcher/<批次号>/` 缓存目录。
- 自动匹配是后台单任务队列，同一进程不会并发执行多个批次。
- `UNMATCHED` 表示源文件存在但算法拒绝归属；只有源分类确实不存在时，网页才显示“文件缺失”。
- 人工确认后的有效归属不会被后续自动重跑覆盖；新版算法建议和诊断仍会保留供比较。
