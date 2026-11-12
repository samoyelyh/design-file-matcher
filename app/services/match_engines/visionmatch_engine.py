from .base import MatchEngine
from app.services.visionmatch import VERSION, status


class VisionMatchEngine(MatchEngine):
    name = "VISIONMATCH"
    label = "VisionMatch AI · Terra"

    def config_snapshot(self):
        return {"engine": self.name, "version": VERSION, "model": "gpt-6.1-terra",
                "mode": "CONTACT_SHEET", "matrix_complete": False, "requires_masks": False}

    def status(self):
        return status()

    def build_score_matrices(self, context):
        raise ValueError("VisionMatch 使用独立拼图流程，不生成虚构相似度矩阵")
