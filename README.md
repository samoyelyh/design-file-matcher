# 设计文件自动匹配、自动命名与网页审核系统

本地版完整流程为：扫描目录并跳过白底图 → 为主图和每个普通图片文件夹手动框选、按需手绘 Mask → 比较“附图手动遮罩 pHash”和“主图手动遮罩 pHash” → PSD 按现有同名文件直接关联 → 按图片分类执行 Hungarian 全局一对一匹配 → 聚合审核 → 生成 Rename Plan。当前匹配路径不使用 DINO、CLIP、自动主体或整图特征参与分配。

## 支持的实际目录结构

系统会递归寻找唯一的 `主图` 目录，因此同时支持简单扁平结构和当前实际交付结构：

```text
项目文件夹/
├── psd/
└── 项目名-效果图/
    ├── 主图/
    ├── 双人模特图/
    ├── 白底模特图/       # 跳过，不索引、不匹配、不改名
    ├── 细节展示图/
    ├── 聚会场景/
    └── 观赛图/
```

选择最外层的“项目文件夹”即可。PSD 可以与效果图目录同级，不要求和主图处在同一层。

## 设计原则

- 图片用途只由一级文件夹决定，不用 AI 猜分类。
- 同一分类内，一张图片最多归属一个商品，一个商品最多获得一张自动匹配图片。
- Hungarian Assignment 使用 Dummy 槽允许拒绝低质量分配，不为了配满而强行归属。
- PSD 不读取图像内容、不计算 pHash，也不进入 Hungarian 分配；系统按 PSD 文件名与主图 basename 直接关联，不同名的 PSD 会安全标记为未匹配。
- 名称包含“白底”的图片目录不进入数据库任务、不参与审核或改名；共享盘源目录和源文件保持原样。可通过 `DESIGN_MATCHER_IGNORED_IMAGE_CATEGORIES` 调整关键词。
- “主图”与其它图片分类并列显示；任选一张主图设置一次裁剪与 Mask，系统按归一化比例批量应用到当前批次全部主图。
- 每个普通图片文件夹（包括细节图）都必须手动设置裁剪与 Mask。匹配只比较两端的手动遮罩裁剪，不计算整图基线，也不再使用自动局部区域作为分配输入。
- 聚合审核支持全选当前结果、通过已选和全部通过当前筛选。整组、单文件或批量确认后会立即刷新 Rename Plan，避免确认状态与改名计划不同步。
- 自动匹配只更新数据库归属并生成 Rename Plan，不移动、删除、覆盖或重命名源文件。Rename Plan 存在待确认、文件缺失或目标冲突时整批禁止执行。
- SQLite 保存原路径、相似度、Top 5 候选、行列 margin、分配排名、算法版本、运行记录和人工覆盖。
- `.design_matcher/` 仅缓存缩略图、预览、Mask 和裁剪图；安全改名全部完成后会自动删除。pHash 保存在 SQLite，源文件未变化时不重复计算。

## 环境与安装

推荐 Python 3.11–3.13。本版不需要 PyTorch、CUDA 或模型权重。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
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
`http://192.168.0.31:8088`。启动脚本会创建隐藏的后台守护进程，命令窗口关闭后
服务仍会运行；网页进程异常退出时会自动重启。使用 `stop-lan.cmd` 可安全停止服务。
运行日志保存在 `%LOCALAPPDATA%\DesignMatcher\service`。pHash 无模型下载步骤。

如果电脑以后连接到其他 Wi-Fi，`192.168.0.31` 可能变化。运行 `ipconfig` 查看新的
IPv4 地址，并继续使用端口 `8088` 即可。

如果 Windows 防火墙以后被启用，需要以管理员身份为 TCP 8088 添加入站允许规则。

可配置项：

```powershell
$env:DESIGN_MATCHER_LOW_CONFIDENCE="0.80"
$env:DESIGN_MATCHER_MIN_SCORE="0.80"
$env:DESIGN_MATCHER_MIN_MARGIN="0.01"
$env:DESIGN_MATCHER_UNMATCHED_SCORE="0.50"
$env:DESIGN_MATCHER_IGNORED_IMAGE_CATEGORIES="白底"
```

匹配处理页会实时显示手动遮罩裁剪、pHash、全局分配和诊断保存进度。

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

测试覆盖白底目录跳过、主图与附图手动遮罩、Top5 持久化、Hungarian、Dummy 拒绝、分类内唯一和人工结果保护。

真实数据的新旧算法对照可执行：

```powershell
python -m scripts.evaluate_phash_phase12 --batch-id 22 --output reports/phase12-batch22.json
```

## 数据与安全

- 数据库默认位于 `data/matcher.db`。
- 每个设计根目录下会生成 `.design_matcher/<批次号>/` 缓存目录。
- 自动匹配是后台单任务队列，同一进程不会并发执行多个批次。
- `UNMATCHED` 表示源文件存在但算法拒绝归属；只有源分类确实不存在时，网页才显示“文件缺失”。
- 人工确认后的有效归属不会被后续自动重跑覆盖；新版算法建议和诊断仍会保留供比较。
