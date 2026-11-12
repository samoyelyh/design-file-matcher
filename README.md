# 设计文件自动匹配、自动命名与网页审核系统

本地版流程为：扫描主图、副图和 PSD → 按文件夹通过 VisionMatch AI · Terra 匹配→ 人工审核 → 只为已确认项目执行安全改名。PSD 使用保存的合成预览参与匹配，不修改图层内容。

## 支持的实际目录结构

系统会递归寻找唯一的 `主图` 目录，因此同时支持简单扁平结构和当前实际交付结构：

```text
项目文件夹/
├── psd/
└── 项目名-效果图/
    ├── 主图/
    ├── 双人模特图/
    ├── 白底模特图/       # 正常索引、匹配、审核和命名
    ├── 细节展示图/
    ├── 聚会场景/
    └── 观赛图/
```

选择最外层的“项目文件夹”即可。PSD 可以与效果图目录同级，不要求和主图处在同一层。

## 设计原则

- 图片用途只由一级文件夹决定，不用 AI 猜分类。
- 同一分类内，一张图片最多归属一个商品，一个商品最多获得一张自动匹配图片。
- Hungarian Assignment 使用 Dummy 槽允许拒绝低质量分配，不为了配满而强行归属。
- PSD 纳入扫描、预览、匹配、审核和已确认项目改名，通过 VisionMatch 匹配，无需遮罩。优先读取 Photoshop 保存的合成预览，并缓存未变化文件的预览。
- 默认纳入全部图片目录和 PSD，包括白底模特图和白底图。只有显式设置 `DESIGN_MATCHER_IGNORED_IMAGE_CATEGORIES` 时才按关键词排除目录；审核页也可临时忽略问题类型。
- “主图”与其它图片分类并列显示；任选一张主图设置一次裁剪与 Mask，系统按归一化比例批量应用到当前批次全部主图。
- 新批次使用整图或 PSD 合成预览执行 Terra 匹配，无需设置识别区域。旧批次可继续使用已有识别区域和匹配记录进行审核。
- 聚合审核支持全选当前结果、通过已选和全部通过当前筛选。整组、单文件或批量确认后会立即刷新 Rename Plan，避免确认状态与改名计划不同步。
- 自动匹配只更新数据库归属并生成 Rename Plan，不移动、删除、覆盖或重命名源文件。命名只处理已人工确认有效归属的文件；未确认项保持原名，不再阻止已确认项执行。已确认范围内的文件缺失或目标冲突仍须处理。
- 审核页“本批次图片类型”可忽略/恢复问题类型；忽略只跳过该批次的审核和改名，保留源图片、已有匹配和审核记录，无需删除批次或重新调用模型。部分改名后保留临时数据，便于继续审核。
- SQLite 保存原路径、相似度、Top 5 候选、行列 margin、分配排名、算法版本、运行记录和人工覆盖。
- `.design_matcher/` 仅缓存缩略图、预览、Mask 和裁剪图；安全改名全部完成后会自动删除。pHash 保存在 SQLite，源文件未变化时不重复计算。

## 环境与安装

推荐 Python 3.11–3.13。本版不需要 PyTorch、CUDA 或模型权重。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
pip install -r .\visionmatch-backend\requirements.txt
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
`http://100.100.0.77:8088`。启动脚本会创建隐藏的后台守护进程，命令窗口关闭后
服务仍会运行；网页进程异常退出时会自动重启。使用 `stop-lan.cmd` 可安全停止服务。
运行日志保存在 `%LOCALAPPDATA%\DesignMatcher\service`。

如果电脑切换网络，使用启动脚本输出的 `LAN` 地址访问。也可以运行 `ipconfig` 查看当前
联网网卡的 IPv4 地址，端口为 `8088`。

如果 Windows 防火墙以后被启用，需要以管理员身份为 TCP 8088 添加入站允许规则。

需要登录 Windows 后自动常驻时运行一次：

```powershell
.\install-autostart.cmd
```

计划任务会隐藏启动守护进程；服务异常退出后守护进程会自动重启它。

通过环境变量 `DESIGN_MATCHER_UPLOAD_ROOT` 设置公共盘根目录。使用 `DESIGN_MATCHER_IGNORED_IMAGE_CATEGORIES` 可按关键词排除不参与匹配的目录。

## 测试

```powershell
pytest
```

测试覆盖白底目录默认纳入及显式排除、主图与附图手动遮罩、Top5 持久化、Hungarian、Dummy 拒绝、分类内唯一和人工结果保护。

真实数据的新旧算法对照可执行：

```powershell
python -m scripts.evaluate_phash_phase12 --batch-id 22 --output reports/phase12-batch22.json
```

多 Reference 的主图、+1、+3、+5 A/B 对照可执行：

```powershell
python -m scripts.evaluate_product_references --truth-status corrected --output data/reference-ab.json
```

评估会排除 Query 自身以及 pHash 完全相同的历史副本，输出整体和各文件夹分类的
Top1 Accuracy、Top3 Recall、Top5 Recall。

## 数据与安全

### VisionMatch AI / Terra 接入（8088）

自动匹配统一使用 VisionMatch AI · Terra，每个副图目录独立处理。
8088 通过本机 `http://127.0.0.1:8066` 调用现有 VisionMatch V2 的导入、队列和结果接口；
不调用它的命名接口。VisionMatch 后端不可用时，引擎提示“后端未启动或未就绪”。

首次配置 Terra 后台常驻与登录自启：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_visionmatch_autostart.ps1
```

该命令注册并启动独立任务 `DesignMatcher-VisionMatch-Service`，不重启8088。
重启电脑并登录 Windows 后，8088与8066分别自动拉起；不是登录前的系统服务。
后端源码位于 `visionmatch-backend/`。守护脚本优先使用已部署的同级 `VisionMatch AI` 目录，否则启动仓库内的后端。仓库内后端默认使用项目的 `.venv`，也可通过 `VISIONMATCH_PYTHON` 指定 Python。后端异常退出后自动重启，数据保存在实际后端目录的 `runtime/v2/`。
日志位于 `%LOCALAPPDATA%\DesignMatcher\visionmatch-service`。
手动启动已安装的任务可执行 `Start-ScheduledTask -TaskName DesignMatcher-VisionMatch-Service`。
仍可使用 VisionMatch 项目的 `start-v2.cmd`；守护脚本检测到8066已占用时不会终止已有进程。

匹配通过已登录 ChatGPT 的 Codex CLI 调用 `gpt-6.1-terra`。
每分类一次原图拼图调用，启动前会列出调用次数并要求确认预算，不自动重试或追加原图复核调用。
每分类可用主图与待识别图片各限 1～50 张，超过限制明确拒绝，不截断候选。
匹配直接使用原图或 PSD 合成预览。

同名直接关联和人工确认图片保留，并为后续识别保留对应主图占位。
输入副本、身份映射、运行ID保存在 `data/visionmatch/<批次号>/`；后端调用日志与缓存仍在 VisionMatch 自己的数据目录。
写回前校验图片ID、内容哈希和分类内唯一归属，失败保留原关联。
VisionMatch 只返回候选、证据与不确定判断，没有全量相似度矩阵；不填充虚构 Top1/Top2 分数。
每次新结果写回时，同分类未匹配图片会全局分配到尚未占用的主图空位：模型备选优先，全部剩余主图都可参与补位。
已有模型关联、同名关联和人工确认占位保留。补位结果标为“自动补位，需复核”，保留原模型判断且不生成相似度。
图片数量足够时填满可用位置；数量不足不复制图片，源图多于空位时保留余图。此规则不在启动或更新时修改历史批次。
审核仍在8088完成，命名预览和实际执行继续使用8088的安全命名流程。

暂停/停止仅影响后续调用，已发出的请求仍可能消耗额度；进程重启或连接中断后不隐式重发模型请求。
单次真实集成探针需显式运行 `scripts/smoke_visionmatch_adapter.py --source 图片路径 --execute-one-call`，
它使用独立测试数据库和一次调用，不能作为批量准确率验收。

- 数据库默认位于 `data/matcher.db`。
- 每个设计根目录下会生成 `.design_matcher/<批次号>/` 缓存目录。
- 自动匹配是后台单任务队列，同一进程不会并发执行多个批次。
- `UNMATCHED` 表示源文件存在但算法拒绝归属；只有源分类确实不存在时，网页才显示“文件缺失”。
- 人工确认后的有效归属不会被后续自动重跑覆盖；新版算法建议和诊断仍会保留供比较。
