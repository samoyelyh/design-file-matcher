from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import provider
from .report import export_report, human_decision
from .concurrent_runner import preview, run
from .execution import ExecutionOptions
from .queue_store import control
from .scan import cli_version, make_plan
from .settings import Settings
from .store import create_run, read_plan


def display_plan(plan: dict) -> None:
    print(f'A 主图：{len(plan["images_a"])} 张；B 候选图：{len(plan["images_b"])} 张')
    print(f'逐对比较：{plan["pair_count"]} 次；Top2 复核上限：{plan["review_max"]} 次')
    print(f'基础 CLI 调用上限：{plan["base_calls_max"]} 次；含有限重试：最多 {plan["max_cli_attempts_with_retries"]} 次')
    print(f'输出目录：{plan["settings"]["output_folder"]}')
    print(f'模型：{plan["settings"]["model"]}；CLI：{plan["cli_version"]}')
    options = ExecutionOptions.from_settings(plan['settings'])
    print(f'并发：{options.concurrency}；硬上限 4；调用预算：{options.max_model_calls}；重试：{options.max_retries}')
    print(f'计划 ID：{plan["plan_id"]}')
    print("CLI 内部网络重试无法保证逐次计数；订阅额度为共享额度。")


def add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.json", help="配置文件")
    parser.add_argument("--limit-a", type=int, help="仅取排序后的前 N 张主图用于小样本")
    parser.add_argument("--limit-b", type=int, help="仅取排序后的前 N 张候选图用于小样本")
    parser.add_argument("--fresh", action="store_true", help="不读取或写入跨任务缓存，适用于独立基准")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="VisionMatch AI 本地图片匹配")
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="只读检查登录和 CLI 版本")
    doctor.add_argument("--config", default="config.json")
    plan = sub.add_parser("plan", help="扫描并显示预计调用量，不运行模型")
    add_plan_args(plan)
    prepare = sub.add_parser("prepare", help="创建独立任务，不运行模型")
    add_plan_args(prepare)
    for command in ('run', 'resume', 'queue-plan'):
        execute = sub.add_parser(command, help="运行/恢复任务或只读查看执行计划")
        execute.add_argument("--run-dir", required=True)
        execute.add_argument("--max-calls", "--max-model-calls", dest="max_calls", type=int, help="累计调用预算，含失败与重试")
        execute.add_argument("--concurrency", type=int, choices=(1,2,3,4))
        execute.add_argument("--max-concurrency", type=int, choices=(1,2,3,4))
        execute.add_argument("--max-retries", type=int, choices=(0,1,2))
        execute.add_argument("--request-timeout", type=float)
        execute.add_argument("--retry-backoff", type=float)
        execute.add_argument("--approve-plan", help="开始真实请求前必须提供计划 ID")
        execute.add_argument("--retry-errors", action="store_true", help="显式重试尚有重试余量的失败项")
    for command in ('pause','stop','status'):
        command_parser = sub.add_parser(command, help="暂停领取、安全停止或查看进度")
        command_parser.add_argument('--run-dir',required=True)
    report = sub.add_parser("report", help="生成新的 CSV、JSON 和 HTML 报告")
    report.add_argument("--run-dir", required=True)
    review = sub.add_parser("review", help="记录人工确认，并生成新的报告")
    review.add_argument("--run-dir", required=True)
    review.add_argument("--a", type=int, required=True, help="从 1 开始的主图编号")
    review.add_argument("--b", type=int, help="从 1 开始的候选图编号；确认未匹配时不填写")
    review.add_argument("--decision", choices=("MATCH", "UNMATCHED"), required=True)
    review.add_argument("--note", default="")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            print(f"Codex 版本：{cli_version()}")
            print(f"登录方式：{provider.login_status()}")
            settings = Settings.load(Path(args.config))
            display_plan(make_plan(settings))
            print("图片、Schema 和模型是否实际可用需要受控真实请求验证。")
        elif args.command in ("plan", "prepare"):
            settings = Settings.load(Path(args.config))
            plan = make_plan(settings, args.limit_a, args.limit_b, fresh=args.fresh)
            display_plan(plan)
            if args.command == "prepare":
                run_dir = create_run(plan)
                print(f"已创建任务：{run_dir}")
                print(f'待执行量：python -m visionmatch queue-plan --run-dir "{run_dir}"')
                print(f'小样本命令：python -m visionmatch run --run-dir "{run_dir}" --max-calls 4 --concurrency 2 --approve-plan {plan["plan_id"]}')
                print(f'按配置预算运行：python -m visionmatch run --run-dir "{run_dir}" --max-calls {plan["settings"].get("max_model_calls",4)} --approve-plan {plan["plan_id"]}')
                print("创建任务未启动模型。请先核对数量及额度并确认。")
        elif args.command in ('run','resume','queue-plan'):
            run_dir = Path(args.run_dir).resolve()
            plan = read_plan(run_dir)
            overrides = {name:getattr(args,name) for name in ('max_concurrency','max_retries','request_timeout','retry_backoff')}
            options = ExecutionOptions.from_settings(plan['settings'], **overrides,
                concurrency=args.concurrency,max_model_calls=args.max_calls)
            print(json.dumps(preview(run_dir, options),ensure_ascii=False,indent=2))
            if args.command=='queue-plan':
                return 0
            print('开始前请确认当前模型已按你的要求切换；此命令将使用计划中的 gpt-6.1-terra。')
            counts = run(run_dir, args.max_calls, concurrency=args.concurrency, approval=args.approve_plan,
                resume=args.command=='resume', retry_errors=args.retry_errors, execution_overrides=overrides)
            print(json.dumps(counts, ensure_ascii=False, indent=2))
            if counts['complete']:
                report_dir = export_report(run_dir)
                print(f"报告：{report_dir / 'index.html'}")
            else:
                print('任务未完整完成，未自动导出最终配对；可使用 status 查看或 report 生成明确标注的阶段报告。')
        elif args.command in ('pause','stop'):
            run_dir=Path(args.run_dir).resolve();read_plan(run_dir)
            control(run_dir, 'PAUSE' if args.command=='pause' else 'STOP')
            print('已请求暂停领取；在途请求完成后退出。' if args.command=='pause' else '已请求安全停止；本调度器的在途进程将被结束。')
        elif args.command=='status':
            from .store import connect
            run_dir=Path(args.run_dir).resolve();read_plan(run_dir)
            db=connect(run_dir/'run.sqlite3')
            try:
                latest=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='queue_sessions'").fetchone()
                finished=db.execute('SELECT finished FROM queue_sessions ORDER BY started DESC LIMIT 1').fetchone() if latest else None
                key='last_queue_status' if finished and finished[0] is not None else 'live_queue_status'
                row=db.execute('SELECT value FROM run_state WHERE key=?',(key,)).fetchone()
                print(row[0] if row else '尚未启动并发调度器。')
            finally:
                db.close()
        elif args.command == "report":
            report_dir = export_report(Path(args.run_dir).resolve())
            print(f"报告：{report_dir / 'index.html'}")
        elif args.command == "review":
            run_dir = Path(args.run_dir).resolve()
            human_decision(run_dir, args.a - 1, args.b - 1 if args.b is not None else None, args.decision, args.note)
            print(f"报告：{export_report(run_dir) / 'index.html'}")
        return 0
    except (ValueError, OSError, RuntimeError, provider.ProviderError) as exc:
        print(f"错误：{provider.scrub(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
