# Phase 0-2 实施与验收报告

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

   增加主图主体 Mask、主体裁剪、主体 pHash 和人工 ROI。

## Phase 1：主图主体遮罩 / 主体 pHash

1. 本阶段完成内容

   已实现近白背景前景提取、形态学去噪、最大连通主体、5% padding、Mask/裁剪保存、整图/主体 pHash 缓存、网页三图对照与矩形 ROI 重选。

2. 修改文件

   `app/services/main_body.py`、`app/services/phash.py`、`app/services/scanner.py`、`app/main.py`、`app/static/index.html`、`app/static/app.js`、`app/static/styles.css`、`tests/test_phash_body.py`、`tests/test_api.py`。

3. 数据库变更

   `main_images` 新增主体 bbox、Mask/裁剪路径、方法、源文件指纹和版本字段；复用 `main_image_features` 保存 `FULL_PHASH`/`BODY_PHASH`。

4. 新增配置

   `DESIGN_MATCHER_BODY_CROP_PADDING=0.05`、`DESIGN_MATCHER_BODY_BACKGROUND_THRESHOLD=245`、`DESIGN_MATCHER_BODY_MIN_COMPONENT_RATIO=0.002`、`DESIGN_MATCHER_BODY_PHASH_VERSION=main_body_phash_v1`。

5. 算法具体逻辑

   RGB 任一通道低于白底阈值即视为前景，闭运算填小缝、开运算去小噪点，保留最大连通区，求 bbox 后按最大边增加 5% padding。前景以外置白后计算 64-bit DCT pHash。人工 ROI 使用归一化坐标，修改后只重算该主图。

6. 测试方式

   合成白底图验证自动 bbox、资产保存、特征缓存和人工 ROI；真实批次 22 抽查原图/Mask/裁剪图，并对 120 张已确认附图做离线对照。

7. 测试结果

   批次 22 的 24/24 张主图成功自动提取，抽查可正确包住两件球衣主体。但整体 Top1 准确率为 5.00%，改用主体 pHash 仍为 5.00%；批次 16 从 7.81% 降至 7.29%。因此“主体提取功能正常”已验证，但“主体 pHash 普遍提升跨场景匹配”未被真实数据证明。

8. 当前风险

   浅色主体、非纯白背景和阴影可能导致 Mask 漏选；pHash 不擅长大幅度构图变换、模特/场景变换，主体裁剪不能单独解决。

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

   细节图计算整图 64-bit pHash，遍历每张主图 11 个区域，使用 `min(hamming_distance)` 作为该主图的局部距离，并转换为 `1 - distance/64` 交给现有分类内 Hungarian 分配器。普通附图改为现有手绘 Masked Crop 的 pHash，PSD 改为 composite preview 纯 pHash；当前活跃匹配路径不下载或调用 CLIP/DINO。

6. 测试方式

   合成局部裁剪图验证最佳主图、最佳 region 和端到端路由；用批次 22（120 个 Ground Truth）和批次 16（192 个 Ground Truth）执行离线评估，不触发改名。

7. 测试结果

   自动化测试 51 项全部通过。批次 22 的细节图整图 pHash 为 4.17%，局部 pHash 为 8.33%，提升 4.17 个百分点；批次 16 从 3.13% 降为 0%。结论是局部逻辑在某些数据有提升，但不稳定，还不达到直接开发 Phase 3 的验收条件。

8. 当前风险

   真实细节图是四宫格文案+多局部拼图，整张 pHash 仍被版式和文字主导；固定主图网格无法保证对齐每个拼图子块。批次 22 的旧细节遮罩框选了整个四宫格，不适合当作单个局部 pHash。

9. 下一阶段计划

   暂不进入 Reference Set。建议先在 Phase 2 内增加“细节图手动选择一个子块 ROI”或轻量固定分块候选，再用这两个批次重新评估；只有准确率稳定上升后再开始 Phase 3。
