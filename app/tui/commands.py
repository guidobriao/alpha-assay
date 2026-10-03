from __future__ import annotations

"""Slash command metadata for the TUI — fully in Chinese."""

from typing import NamedTuple


class CommandMeta(NamedTuple):
    name: str
    args: str                  # internal args string (for parsing reference)
    description: str           # Chinese description
    category: str              # Chinese category
    safe_during_run: bool = False
    display_args: str = ""     # Chinese display string for completion popup

    @property
    def has_required_args(self) -> bool:
        """True if the command takes non-optional arguments."""
        return bool(self.args) and not self.args.startswith("[")


COMMANDS: dict[str, CommandMeta] = {
    "help": CommandMeta("help", "", "显示帮助", "系统", True),
    "clear": CommandMeta("clear", "", "清空消息时间线", "系统", True),
    "status": CommandMeta("status", "", "查看当前任务状态", "查看", True),
    "plan": CommandMeta("plan", "", "切换到计划模式，不执行命令", "模式", False),
    "act": CommandMeta("act", "", "切换到执行模式", "模式", False),
    "input": CommandMeta("input", "<path>", "设置本地论文 PDF 路径", "输入", False, display_args="<PDF路径>"),
    "arxiv": CommandMeta("arxiv", "<arXiv ID 或 URL>", "从 arXiv 下载 PDF 并设为当前论文", "输入", False, display_args="<arXiv ID或URL>"),
    "download-arxiv": CommandMeta("download-arxiv", "<arXiv ID 或 URL>", "从 arXiv 下载 PDF 并设为当前论文", "输入", False, display_args="<arXiv ID或URL>"),
    "repo": CommandMeta("repo", "<url>", "直接指定 GitHub 仓库地址", "输入", False, display_args="<仓库URL>"),
    "repo-dir": CommandMeta("repo-dir", "<path>", "指定本地代码仓库目录", "输入", False, display_args="<本地仓库目录>"),
    "backend": CommandMeta("backend", "[none|local|venv|conda|docker]", "设置执行后端", "运行", False, display_args="<后端>"),
    "workspace": CommandMeta("workspace", "<path>", "设置输出工作目录", "运行", False, display_args="<目录>"),
    "timeout": CommandMeta("timeout", "<minutes>", "设置步骤超时时间（分钟）", "运行", False, display_args="<分钟数>"),
    "repairs": CommandMeta("repairs", "<n>", "设置最大依赖修复次数", "运行", False, display_args="<次数>"),
    "run": CommandMeta("run", "", "执行复现流水线", "运行", False),
    "report": CommandMeta("report", "", "查看复现报告", "查看", True),
    "logs": CommandMeta("logs", "<type>", "查看日志（env/smoke/benchmark 等）", "查看", True, display_args="<日志类型>"),
    "cancel": CommandMeta("cancel", "", "取消当前任务", "运行", True),
    "kill": CommandMeta("kill", "", "强制终止当前任务和子进程", "运行", True),
    "force-cancel": CommandMeta("force-cancel", "", "强制终止当前任务和子进程", "运行", True),
    "abort": CommandMeta("abort", "", "强制终止当前任务和子进程", "运行", True),
    "sessions": CommandMeta("sessions", "", "列出历史会话", "会话", True),
    "session": CommandMeta("session", "", "列出历史会话", "会话", True),
    "resume": CommandMeta("resume", "<session-id>", "恢复历史会话", "会话", False, display_args="<会话ID>"),
    "quit": CommandMeta("quit", "", "退出 TUI", "系统", True),
    "exit": CommandMeta("exit", "", "退出 TUI", "系统", True),
    "panel": CommandMeta("panel", "[session|pipeline|help|artifacts|none]", "切换右侧面板", "界面", True, display_args="<面板>"),
    "artifact": CommandMeta("artifact", "", "显示当前任务产物路径", "查看", True),
    "open-report": CommandMeta("open-report", "", "显示报告文件路径", "查看", True),
    "mode": CommandMeta("mode", "", "显示当前 PLAN/ACT 模式", "模式", True),
    "reset": CommandMeta("reset", "", "清空当前会话输入（不删文件）", "会话", False),
    "conda-envs": CommandMeta("conda-envs", "[list|delete|prune]", "查看或删除项目 conda 环境", "环境", True, "[list|delete|prune]"),
    "envs": CommandMeta("envs", "[list|delete|prune]", "查看或删除项目 conda 环境", "环境", True, "[list|delete|prune]"),
}

RUNNING_SAFE_COMMANDS: set[str] = {n for n, m in COMMANDS.items() if m.safe_during_run}
