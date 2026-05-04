"""Git Auto Updater - 최대 3개 프로그램 등록 가능"""
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import os
import json
import shutil
import threading
import sys


class GitAutoUpdater:
    MAX_SLOTS = 3

    def __init__(self, root):
        self.root = root
        self.root.title("Git Auto Updater")
        self.root.geometry("660x740")
        self.root.minsize(500, 600)

        self.config_path = self._config_path()
        self.entries = []

        self._build_ui()
        self._load_config()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._check_git()

    # ── 경로 ──────────────────────────────────────────────

    def _config_path(self):
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "updater_config.json")

    # ── UI ────────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 헤더
        hdr = ttk.Frame(main)
        hdr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(hdr, text="Git Auto Updater", font=("Segoe UI", 14, "bold")).pack(
            side=tk.LEFT
        )
        ttk.Button(hdr, text="전체 업데이트", command=self._update_all).pack(
            side=tk.RIGHT
        )

        # 슬롯 3개
        for i in range(self.MAX_SLOTS):
            self._add_slot(main, i)

        # 로그
        log_frm = ttk.Frame(main)
        log_frm.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        log_hdr = ttk.Frame(log_frm)
        log_hdr.pack(fill=tk.X)
        ttk.Label(log_hdr, text="로그").pack(side=tk.LEFT)
        ttk.Button(log_hdr, text="지우기", width=5, command=self._clear_log).pack(
            side=tk.RIGHT
        )
        self.log_area = scrolledtext.ScrolledText(
            log_frm, height=10, state=tk.DISABLED, font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def _add_slot(self, parent, idx):
        frm = ttk.LabelFrame(parent, text=f" 프로그램 {idx + 1} ", padding=6)
        frm.pack(fill=tk.X, pady=3)
        e = {}

        def make_row(label_text, key, default=""):
            r = ttk.Frame(frm)
            r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=label_text, width=8, anchor=tk.E).pack(side=tk.LEFT)
            entry = ttk.Entry(r)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
            if default:
                entry.insert(0, default)
            e[key] = entry
            return r, entry

        make_row("이름:", "name")

        path_row, path_entry = make_row("경로:", "path")
        ttk.Button(
            path_row,
            text="...",
            width=3,
            command=lambda: self._browse(path_entry),
        ).pack(side=tk.RIGHT)

        make_row("URL:", "url")

        branch_row, _ = make_row("브랜치:", "branch", "main")
        ttk.Button(
            branch_row,
            text="업데이트",
            width=8,
            command=lambda i=idx: self._update_single(i),
        ).pack(side=tk.RIGHT)

        self.entries.append(e)

    # ── Helpers ───────────────────────────────────────────

    def _browse(self, entry):
        p = filedialog.askdirectory()
        if p:
            entry.delete(0, tk.END)
            entry.insert(0, p)

    def _log(self, msg):
        w = self.log_area
        w.config(state=tk.NORMAL)
        w.insert(tk.END, msg + "\n")
        w.see(tk.END)
        w.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete("1.0", tk.END)
        self.log_area.config(state=tk.DISABLED)

    def _thread_log(self, msg):
        self.root.after(0, self._log, msg)

    def _slot_data(self, i):
        return {
            k: self.entries[i][k].get().strip() for k in ("name", "path", "url", "branch")
        }

    # ── Config ────────────────────────────────────────────

    def _save_config(self):
        data = [self._slot_data(i) for i in range(self.MAX_SLOTS)]
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self._log(f"설정 저장 실패: {exc}")

    def _load_config(self):
        if not os.path.isfile(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for i, item in enumerate(items[: self.MAX_SLOTS]):
                for key in ("name", "path", "url", "branch"):
                    val = item.get(key, "")
                    if val:
                        self.entries[i][key].insert(0, val)
        except Exception:
            pass

    def _on_close(self):
        self._save_config()
        self.root.destroy()

    # ── Git ───────────────────────────────────────────────

    def _check_git(self):
        try:
            subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        except Exception:
            self._log("[경고] git을 찾을 수 없습니다. 설치 여부를 확인하세요.")

    def _clean_cache(self, path):
        """__pycache__ 디렉토리 및 .pyc/.pyo 파일 삭제"""
        removed = 0
        for dirpath, dirs, files in os.walk(path):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                try:
                    shutil.rmtree(os.path.join(dirpath, "__pycache__"))
                    removed += 1
                except Exception:
                    pass
                dirs.remove("__pycache__")
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    try:
                        os.remove(os.path.join(dirpath, f))
                        removed += 1
                    except Exception:
                        pass
        return removed

    def _verify(self, path, branch, name):
        """로컬 HEAD와 원격 브랜치 커밋 해시 + 작업 트리 상태 검증"""
        # 커밋 해시 비교
        local = self._git(["git", "rev-parse", "HEAD"], path)
        remote = self._git(["git", "rev-parse", f"origin/{branch}"], path)
        local_hash = local.stdout.strip()
        remote_hash = remote.stdout.strip()

        if not local_hash or not remote_hash:
            self._thread_log(f"[{name}] 검증 실패 - 커밋 해시 조회 불가")
            return False

        if local_hash != remote_hash:
            self._thread_log(f"[{name}] 검증 실패 - 로컬과 원격이 다름")
            self._thread_log(f"  로컬: {local_hash[:8]}  원격: {remote_hash[:8]}")
            return False

        # 작업 트리 변경 사항 확인
        status = self._git(["git", "status", "--porcelain"], path)
        changes = [l for l in status.stdout.strip().splitlines() if l.strip()]
        if changes:
            self._thread_log(f"[{name}] 검증 실패 - 로컬 변경사항 {len(changes)}건 감지")
            for c in changes[:5]:
                self._thread_log(f"  {c}")
            if len(changes) > 5:
                self._thread_log(f"  ... 외 {len(changes) - 5}건")
            return False

        self._thread_log(f"[{name}] 검증 완료 - {local_hash[:8]}")
        return True

    def _git(self, args, cwd, timeout=120):
        return subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )

    def _do_update(self, data):
        name = data["name"] or os.path.basename(data["path"]) or "Program"
        path = data["path"]
        url = data["url"]
        branch = data["branch"] or "main"

        if not path:
            self._thread_log(f"[{name}] 경로가 비어 있습니다.")
            return
        if not url:
            self._thread_log(f"[{name}] URL이 비어 있습니다.")
            return

        git_dir = os.path.join(path, ".git")

        try:
            if os.path.isdir(git_dir):
                # ── 기존 repo: 캐시 정리 → pull ──
                self._thread_log(f"[{name}] 캐시 정리 중...")
                cnt = self._clean_cache(path)
                self._thread_log(f"[{name}] 캐시 {cnt}개 삭제됨")

                # 원격 URL 변경 감지
                r = self._git(["git", "remote", "get-url", "origin"], path)
                if r.stdout.strip() != url:
                    self._git(["git", "remote", "set-url", "origin", url], path)
                    self._thread_log(f"[{name}] 원격 URL 갱신됨")

                # git pull
                self._thread_log(f"[{name}] git pull origin {branch} ...")
                r = self._git(["git", "pull", "origin", branch], path)
                if r.returncode == 0:
                    if "Already up to date" in r.stdout:
                        self._thread_log(f"[{name}] 이미 최신 상태입니다.")
                    else:
                        self._thread_log(f"[{name}] 업데이트 완료!")
                    # 유효성 검사
                    self._git(["git", "fetch", "origin"], path)
                    self._verify(path, branch, name)
                else:
                    err = (r.stderr or r.stdout).strip()
                    if "CONFLICT" in err or "merge" in err.lower():
                        self._thread_log(f"[{name}] 업데이트 실패 - 병합 충돌 발생")
                    else:
                        self._thread_log(f"[{name}] 업데이트 실패")
                    self._thread_log(f"[{name}] 사유: {err}")
            else:
                # ── .git 없음 → clone 또는 init ──
                os.makedirs(path, exist_ok=True)
                if not os.listdir(path):
                    # 빈 디렉토리: clone
                    self._thread_log(f"[{name}] git clone -b {branch} ...")
                    r = self._git(["git", "clone", "-b", branch, url, "."], path, timeout=300)
                else:
                    # 파일 있는 디렉토리: init → fetch → checkout
                    self._thread_log(f"[{name}] git 초기화 + 동기화 ...")
                    self._git(["git", "init"], path)
                    self._git(["git", "remote", "add", "origin", url], path)
                    self._git(["git", "fetch", "origin"], path, timeout=300)
                    r = self._git(
                        ["git", "checkout", "-f", "-b", branch, f"origin/{branch}"], path
                    )

                if r.returncode == 0:
                    self._thread_log(f"[{name}] 완료!")
                    self._verify(path, branch, name)
                else:
                    self._thread_log(f"[{name}] 실패: {r.stderr.strip()}")

        except subprocess.TimeoutExpired:
            self._thread_log(f"[{name}] 시간 초과!")
        except Exception as exc:
            self._thread_log(f"[{name}] 오류: {exc}")

    def _update_single(self, idx):
        self._save_config()
        data = self._slot_data(idx)
        threading.Thread(target=self._do_update, args=(data,), daemon=True).start()

    def _update_all(self):
        self._save_config()

        def runner():
            for i in range(self.MAX_SLOTS):
                d = self._slot_data(i)
                if d["path"]:
                    self._do_update(d)
            self._thread_log("=== 전체 업데이트 완료 ===")

        threading.Thread(target=runner, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    GitAutoUpdater(root)
    root.mainloop()
