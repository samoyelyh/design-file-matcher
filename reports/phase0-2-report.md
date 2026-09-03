# Phase 0-2 实施与验收报告

> 2026-09-03 业务调整：以下报告保留历史评估结论，但当前活跃匹配路径已经改为“主图手动遮罩 pHash vs 附图手动遮罩 pHash”。扫描不再自动生成主图整图特征，细节图不再绕过手动遮罩走固定局部区域，名称包含“白底”的目录不索引、不匹配、不改名，PSD 仅按同名主图直连。

## Phase 0：现有项目审查

1. 本阶段完成内容

   已确认扫描入口、pHash/Hamming Distance、分类内全局分配、安全改名、SQLite 模型、网页审核、PSD composite preview 和恢复日志的位置。

2. 修改文件

   Phase 0 本身只审查，没有为它单独修改业务文件。

3. 数据库变更

   无。

4. 新增配置

   无。

5. 算法具体逻辑

   可复用 `scanner.py` 的目录分类、`assignment.py` 的 Hungarian 分类内唯一分配、`rename_plan.py` 的两阶段安全改名与清理、现有聚合审核与人工调整。不改文件夹分类规则、主图 basename 命名规则、分类内唯一约束和禁止覆盖约束。

6. 测试方式

   阅读模型、服务层、API、前端和现有测试，并检查最近真实批次的数据量、分类和人工确认结果。

7. 测试结果

   批次 22 可直接访问共享盘，24 张主图、120 张已确认附图，可作为 Ground Truth。批次 16 含 32 张主图、192 张已确认文件。

8. 当前风险

   旧代码实际仍有 OpenCLIP 活跃路径，与本次文档的“纯 pHash”前提不一致；本轮已将匹配入口改为纯 pHash，保留旧数据库 embedding 字段只为兼容历史数据。

9. 下一阶段计划

   主图默认保留整图白底，并与其它图片分类并列；裁剪和手绘遮罩设置一次后按比例批量应用到全部主图。

## Phase 1：主图主体遮罩 / 主体 pHash

1. 本阶段完成内容

   主图已作为普通“图片分类”进入统一裁剪与手绘 Mask 编辑器。任选一张参考主图设置一次后，归一化裁剪框和遮罩会按比例批量应用到全部主图，并重算整图、裁剪图和局部 pHash。系统不再自动去除白底。

2. 修改文件

   `app/services/main_body.py`、`app/services/phash.py`、`app/services/scanner.py`、`app/main.py`、`app/static/index.html`、`app/static/app.js`、`app/static/styles.css`、`tests/test_phash_body.py`、`tests/test_api.py`。

3. 数据库变更

   `main_images` 新增主体 bbox、Mask/裁剪路径、方法、源文件指纹和版本字段；复用 `main_image_features` 保存 `FULL_PHASH`/`BODY_PHASH`。

4. 新增配置

   `DESIGN_MATCHER_BODY_PHASH_VERSION=main_body_phash_v2_full_image`，用于让旧的自动去白底缓存升级为默认保留整图。

5. 算法具体逻辑

   未设置主图 Mask Profile 时使用全白遮罩和完整画面。用户设置后，系统把同一个归一化裁剪框和遮罩缩放到每张主图尺寸，裁剪框外置黑、框内按手绘白黑区域保留或排除，再批量计算 64-bit DCT pHash。

6. 测试方式

   合成白底图验证默认整图、资产保存和特征缓存；两张主图的 API 测试验证一次保存会批量应用相同归一化裁剪框和遮罩。

7. 测试结果

   自动去白底的历史离线对照没有带来稳定收益，因此当前版本按业务要求移除该默认行为，改由人工按分类设置一次并批量应用。

8. 当前风险

   所有主图共用同一归一化区域，因此要求版式大体一致；如果某张主图版式明显不同，需要先统一素材版式或建立新批次。

9. 下一阶段计划

   保留主体功能作为 Phase 2 局部区域的基础，不将主体 pHash 强行替代所有整图证据。

## Phase 2：细节图局部识别

1. 本阶段完成内容

   已生成 `FULL`、`BODY`、`CENTER`、`TOP`、`BOTTOM`、`LEFT`、`RIGHT`、`TOP_LEFT`、`TOP_RIGHT`、`BOTTOM_LEFT`、`BOTTOM_RIGHT` 共 11 种 pHash；细节图对每张主图取最小 Hamming Distance，保存最佳区域和距离。

2. 修改文件

   `app/services/detail_phash.py`、`app/services/phash_matching.py`、`app/services/matcher.py`、`app/services/psd.py`、`app/models.py`、`app/db.py`、`scripts/evaluate_phash_phase12.py`、相关测试和前端诊断展示。

3. 数据库变更

   新增 `image_phash_regions` 表；`matched_files` 增加源 pHash 缓存指纹和 `phash_best_region_type`；`derived_image_assets` 增加 pHash 及源文件指纹缓存字段。

4. 新增配置

   `DESIGN_MATCHER_DETAIL_CATEGORY_KEYWORDS=细节,detail`、`DESIGN_MATCHER_DETAIL_REGION_VERSION=detail_region_phash_v1`。匹配入口版本为 `full_phash_hungarian_v1` / `masked_phash_hungarian_v1`。

5. 算法具体逻辑

   细节图计算整图 64-bit pHash，遍历每张主图 11 个区域，使用 `min(hamming_distance)` 作为该主图的局部距离，并转换为 `1 - distance/64` 交给现有分类内 Hungarian 分配器。普通附图使用手绘 Masked Crop 的 pHash；PSD 不读取内容、不计算相似度，按文件名与主图 basename 直接关联。当前活跃匹配路径不下载或调用 CLIP/DINO。

6. 测试方式

   合成局部裁剪图验证最佳主图、最佳 region 和端到端路由；用批次 22（120 个 Ground Truth）和批次 16（192 个 Ground Truth）执行离线评估，不触发改名。

7. 测试结果

   自动化测试 51 项全部通过。批次 22 的细节图整图 pHash 为 4.17%，局部 pHash 为 8.33%，提升 4.17 个百分点；批次 16 从 3.13% 降为 0%。结论是局部逻辑在某些数据有提升，但不稳定，还不达到直接开发 Phase 3 的验收条件。

8. 当前风险

   真实细节图是四宫格文案+多局部拼图，整张 pHash 仍被版式和文字主导；固定主图网格无法保证对齐每个拼图子块。批次 22 的旧细节遮罩框选了整个四宫格，不适合当作单个局部 pHash。

9. 下一阶段计划

   暂不进入 Reference Set。建议先在 Phase 2 内增加“细节图手动选择一个子块 ROI”或轻量固定分块候选，再用这两个批次重新评估；只有准确率稳定上升后再开始 Phase 3。
