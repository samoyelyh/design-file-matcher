# 设计文件自动匹配、自动命名与网页审核系统

当前流程：扫描公共盘项目目录 → 为每个普通图片文件夹手动画框与遮罩 → 对裁剪区域生成 DINO 视觉特征（PSD 使用整图 Preview）→ 按文件夹独立执行 cosine + Hungarian 一对一匹配 → 按本次手动设置的最低归属分、低置信度分数和最小 Gap 保存 Top5 与判断结果 → 无覆盖自动命名 → 网页按主图整组审核和换队。

## 目录结构

系统递归寻找唯一的 `主图` 目录，图片分类名称直接取实际文件夹名称；PSD 目录可以与效果图目录同级。

```text
项目文件夹/
├── psd/
└── 项目名-效果图/
    ├── 主图/
    ├── 双人模特图/
    ├── 细节展示图/
    └── 任意其他图片文件夹/
```

- 普通图片与主图都使用 DINO 全图 embedding。
- PSD 由 `psd-tools` 合成预览后送入同一个 DINO 模型；pHash 只保留为辅助诊断，不参与最终排序。
- 每个文件夹独立一对一分配；文件数多于主图数时，多余文件保持原名并标记忽略。
- 匹配完成立即执行安全改名。目标已被非本批次文件占用时绝不覆盖，而是标记冲突供网页处理。
- 网页换队遇到已占用球队时会交换归属，并通过临时文件安全互换两个物理文件名。
- 人工确认结果不会被重新匹配覆盖。

## 安装与模型

推荐 Python 3.11：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dino,dev]"
```

默认使用 DINOv2 Base。项目通过南京 HF 镜像下载模型到本地目录，规避新版 `huggingface_hub` 对镜像 HEAD 元数据的兼容问题：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_dinov2_mirror.ps1
```

下载完成后模型位于 `models\dinov2-base`，服务以离线方式加载，不会重复下载。

## 配置

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:EMBEDDING_BACKEND="dinov2"              # 可改为 dinov3
$env:DESIGN_MATCHER_DINOV2_LOCAL_DIR="D:\models\dinov2-base"
$env:DESIGN_MATCHER_DINOV3_MODEL="facebook/dinov3-vitb16-pretrain-lvd1689m"
$env:DESIGN_MATCHER_DINO_ALLOW_FALLBACK="1" # DINOv3 无权限时回退 DINOv2
$env:DESIGN_MATCHER_BATCH_SIZE="8"           # 显存不足会自动减半重试
$env:DESIGN_MATCHER_MIN_SCORE="0.80"
$env:DESIGN_MATCHER_MIN_MARGIN="0.0005"       # DINO 的 Top1/Top2 常非常接近
$env:DESIGN_MATCHER_UNMATCHED_SCORE="0.50"
```

DINOv3 官方模型需要先接受许可并登录；没有权限时建议保持 `dinov2`。系统自动使用 CUDA，显卡不可用时回退 CPU。

## 局域网启动

```powershell
.\start-lan.cmd
```

- 本机：`http://127.0.0.1:8088`
- 局域网：`http://192.168.0.31:8088`（IP 变化时用 `ipconfig` 查询新的 IPv4）
- 公共盘默认根目录：`O:\1-亚马逊数据库\设计库\文件包上传测试`

## 只读质量验证

以下命令只输出 DINO Top5，不建批次、不改名：

```powershell
.\.venv\Scripts\python.exe scripts\verify_match.py "D:\设计目录" --category "观赛图" --limit 30
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app tests scripts
node --check app\static\app.js
```

数据库默认位于 `data/matcher.db`。`.design_matcher/` 只存缩略图、PSD 预览和版本化 embedding；最终整批确认并执行命名后会自动清理。
