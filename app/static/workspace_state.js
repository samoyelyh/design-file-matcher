(function (root) {
  function deriveWorkspaceState(batch, review = {}, plan = {}) {
    if (!batch) {
      return {
        currentStep: "import",
        nextView: "create",
        nextLabel: "导入图片",
        canEnter: { create: true, mask: false, process: false, review: false },
        steps: { import: "current", regions: "upcoming", match: "upcoming", review: "upcoming" },
        totals: { files: 0, regionsDone: 0, regionsTotal: 0, pending: 0, unmatched: 0 },
      };
    }

    const files = Object.values(batch.categories || {}).reduce((sum, value) => sum + Number(value || 0), 0);
    const regionsTotal = 0;
    const regionsDone = 0;
    const imported = Number(batch.main_count || 0) > 0 && files > 0;
    const regionsReady = imported;
    const matching = ["QUEUED", "MATCHING"].includes(batch.status);
    const matched = batch.status === "COMPLETED";
    const failed = batch.status === "FAILED";
    const summary = review.summary || {};
    const pending = Number(summary.pending || 0);
    const unmatched = Number(summary.unmatched_files || 0);
    const blockers = (plan.plans || []).filter(item => !["READY", "NOOP", "RENAMED", "IGNORED"].includes(item.rename_status)).length;
    const needsReview = matched && (pending > 0 || unmatched > 0 || blockers > 0);

    let currentStep = "import";
    let nextView = "create";
    let nextLabel = imported ? "查看导入结果" : "导入图片";
    if (regionsReady && !matched) {
      currentStep = "match";
      nextView = "process";
      nextLabel = matching ? "查看匹配进度" : failed ? "重新开始自动匹配" : "开始自动匹配";
    } else if (matched) {
      currentStep = "review";
      nextView = "review";
      nextLabel = needsReview ? "审核需要确认的图片" : "预览命名结果";
    }

    return {
      currentStep,
      nextView,
      nextLabel,
      matching,
      matched,
      needsReview,
      blockers,
      canEnter: { create: true, mask: false, process: regionsReady, review: matched },
      steps: {
        import: imported ? "complete" : "current",
        regions: imported ? "complete" : "upcoming",
        match: !regionsReady ? "upcoming" : matched ? "complete" : failed ? "needs" : "current",
        review: !matched ? "upcoming" : needsReview ? "needs" : "current",
      },
      totals: { files, regionsDone, regionsTotal, pending, unmatched },
    };
  }

  root.deriveWorkspaceState = deriveWorkspaceState;
  if (typeof module !== "undefined" && module.exports) module.exports = { deriveWorkspaceState };
})(typeof window !== "undefined" ? window : globalThis);
