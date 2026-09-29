"""시연 영상 녹화 — `rec on` 부터 `rec off` 까지 게임 화면을 mp4 한 편으로 남긴다 (`recording`)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from core.config import recording_opt

GB_W, GB_H = 160, 144
VBLANK_FALLBACK = (0, 0x2024)          # pokered.sym "00:2024 VBlank"
GAME_FPS = 60

_BG = (24, 26, 32)
_FG = (232, 234, 238)
_DIM = (150, 156, 168)
_ACCENT = (255, 204, 64)

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)


def _load_font(path: str, size: int):
    from PIL import ImageFont

    for cand in ([path] if path else []) + list(_FONT_CANDIDATES):
        if cand and os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    print("[rec] 한글 글꼴을 못 찾았습니다 — recording.font 에 .ttf 경로를 적으세요 (패널 글자가 깨질 수 있음)")
    return ImageFont.load_default()


def vblank_address(pyboy: Any):
    """VBlank 루틴 주소 (bank, addr)."""
    try:
        bank, addr = pyboy.symbol_lookup("VBlank")
        return int(bank), int(addr)
    except Exception:
        return VBLANK_FALLBACK


def _open_writer(path: Path, fps: int, size) -> Any:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if writer.isOpened():
        return writer, path
    alt = path.with_suffix(".avi")                 # mp4 코덱이 없는 OpenCV 빌드
    writer = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), size)
    return (writer, alt) if writer.isOpened() else (None, path)


class DemoRecorder:
    """``start()`` ~ ``stop()`` 한 번 = 영상 한 편."""

    def __init__(
        self,
        pyboy: Any,
        *,
        speed: int = 8,
        fps: int = GAME_FPS,
        scale: int = 3,
        overlay: bool = True,
        panel_width: int = 360,
        plan_hold_sec: float = 1.0,
        out_dir: str = "recordings",
        font: str = "",
        status: Optional[Callable[[], Sequence[str]]] = None,
        writer_factory: Optional[Callable[[Path, int, tuple], Any]] = None,
    ) -> None:
        self.pyboy = pyboy
        self.speed = max(1, int(speed))
        self.fps = max(1, int(fps))
        # 게임 몇 프레임마다 한 장을 담는가 — 재생 fps 가 바뀌어도 배속이 그대로이게
        self.interval = max(1, int(round(self.speed * GAME_FPS / self.fps)))
        self.scale = max(1, int(scale))
        self.overlay = bool(overlay)
        self.panel_width = int(panel_width) if self.overlay else 0
        self.plan_hold_sec = max(0.0, float(plan_hold_sec))
        self.out_dir = Path(out_dir)
        self.font_path = str(font or "")
        self.status = status
        self._writer_factory = writer_factory or _open_writer
        self._writer = None
        self.path: Optional[Path] = None
        self.active = False
        self.attached = False
        self.frames = 0                 # 녹화 중 지나간 게임 프레임
        self.written = 0                # 영상에 쓴 장 수
        self._last_screen: Optional[np.ndarray] = None
        self._panel_key: Optional[tuple] = None
        self._panel: Optional[np.ndarray] = None
        self._fonts = None
        self._started_at = 0.0

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, pyboy: Any, status: Optional[Callable[[], Sequence[str]]] = None) -> "DemoRecorder":
        """settings/config.yaml 의 ``recording`` 값으로 만든다."""
        return cls(
            pyboy,
            speed=int(recording_opt("speed", 8) or 8),
            fps=int(recording_opt("fps", GAME_FPS) or GAME_FPS),
            scale=int(recording_opt("scale", 3) or 3),
            overlay=bool(recording_opt("overlay", True)),
            plan_hold_sec=float(recording_opt("plan_hold_sec", 1.0) or 0.0),
            out_dir=str(recording_opt("out_dir", "recordings") or "recordings"),
            font=str(recording_opt("font", "") or ""),
            status=status,
        )

    @property
    def size(self) -> tuple:
        return GB_W * self.scale + self.panel_width, GB_H * self.scale

    def attach(self) -> bool:
        """VBlank 훅을 건다 (세션에 한 번)."""
        if self.attached:
            return True
        try:
            import cv2  # noqa: F401
        except Exception:
            print("[rec] opencv-python 이 없어 녹화를 못 합니다 — pip install opencv-python")
            return False
        bank, addr = vblank_address(self.pyboy)
        try:
            self.pyboy.hook_register(bank, addr, self._on_vblank, None)
        except Exception as exc:
            print(f"[rec] VBlank 훅 등록 실패: {type(exc).__name__}: {exc}")
            return False
        self.attached = True
        return True

    def start(self, label: str = "play") -> Optional[Path]:
        if self.active or not self.attach():
            return self.path if self.active else None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{label}_{stamp}_x{self.speed}.mp4"
        n = 2
        while path.exists() or path == self.path:       # 같은 초에 다시 켜도 덮어쓰지 않는다
            path = self.out_dir / f"{label}_{stamp}_x{self.speed}_{n}.mp4"
            n += 1
        self._writer, self.path = self._writer_factory(path, self.fps, self.size)
        if self._writer is None:
            print(f"[rec] 영상 파일을 못 엽니다: {path}")
            return None
        self.frames = 0
        self.written = 0
        self._panel_key = None
        self._started_at = time.time()
        self.active = True
        print(f"[rec] 녹화 시작 → {self.path}  ({self.speed}배속 · {self.fps}fps · {self.size[0]}x{self.size[1]})")
        return self.path

    def status_line(self) -> str:
        if not self.active:
            return f"[rec] 꺼짐 — rec on 으로 시작 ({self.speed}배속 · {self.out_dir}/)"
        secs = self.written / float(self.fps)
        return (f"[rec] 녹화 중 → {self.path}  (영상 {secs / 60:.1f}분 · "
                f"게임 {self.frames / float(GAME_FPS) / 60:.1f}분 · {self.written}장)")

    def stop(self, reason: str = "") -> Optional[Path]:
        if not self.active:
            return None
        self.active = False
        try:
            self._writer.release()
        except Exception:
            pass
        self._writer = None
        secs = self.written / float(self.fps)
        game = self.frames / float(GAME_FPS)
        why = f" — {reason}" if reason else ""
        print(
            f"[rec] 녹화 끝{why} → {self.path}  "
            f"(영상 {secs / 60:.1f}분 · 게임 {game / 60:.1f}분 · {self.written}장)"
        )
        return self.path

    def hold(self, seconds: Optional[float] = None) -> int:
        """지금 화면을 ``seconds`` 초 멈춰 보여 준다 (새 계획이 나온 순간)."""
        if not self.active or self._last_screen is None:
            return 0
        sec = self.plan_hold_sec if seconds is None else max(0.0, float(seconds))
        n = int(round(sec * self.fps))
        if n <= 0:
            return 0
        frame = self._compose(self._last_screen)
        for _ in range(n):
            self._write(frame)
        return n

    # ------------------------------------------------------------------
    def _on_vblank(self, _ctx: Any) -> None:
        if not self.active:
            return
        self.frames += 1
        if self.frames % self.interval:
            return
        try:
            screen = np.array(self.pyboy.screen.ndarray[:, :, :3], dtype=np.uint8, copy=True)
            self._last_screen = screen
            self._write(self._compose(screen))
        except Exception as exc:                 # 녹화 오류로 게임을 멈추지 않는다
            print(f"[rec] 프레임 기록 실패 — 녹화를 멈춥니다: {type(exc).__name__}: {exc}")
            self.stop("오류")

    def _write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self.written += 1

    def _compose(self, screen_rgb: np.ndarray) -> np.ndarray:
        import cv2

        game = cv2.resize(screen_rgb, (GB_W * self.scale, GB_H * self.scale), interpolation=cv2.INTER_NEAREST)
        game = cv2.cvtColor(game, cv2.COLOR_RGB2BGR)
        if not self.overlay:
            return game
        frame = np.hstack([game, self._panel_image()])
        clock = self.frames // GAME_FPS
        text = f"x{self.speed}  {clock // 3600:02d}:{clock // 60 % 60:02d}:{clock % 60:02d}"
        cv2.putText(frame, text, (GB_W * self.scale + 16, GB_H * self.scale - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DIM[::-1], 1, cv2.LINE_AA)
        return frame

    def _status_lines(self) -> List[str]:
        if self.status is None:
            return []
        try:
            return [str(s) for s in (self.status() or [])]
        except Exception as exc:
            return [f"(상태 읽기 실패: {type(exc).__name__})"]

    def _panel_image(self) -> np.ndarray:
        lines = self._status_lines()
        key = tuple(lines)
        if key == self._panel_key and self._panel is not None:
            return self._panel
        self._panel_key = key
        self._panel = render_panel(lines, self.panel_width, GB_H * self.scale, self._get_fonts())
        return self._panel

    def _get_fonts(self):
        if self._fonts is None:
            size = max(12, 6 * self.scale)
            self._fonts = (_load_font(self.font_path, size + 2), _load_font(self.font_path, size))
        return self._fonts


def _wrap(text: str, font: Any, width: int) -> List[str]:
    """글자 단위 줄바꿈 (한글은 띄어쓰기가 드물어 단어 단위로 자르면 넘친다)."""
    out: List[str] = []
    line = ""
    for ch in text:
        if font.getlength(line + ch) <= width:
            line += ch
            continue
        out.append(line)
        line = ch.lstrip() if ch == " " else ch
    if line or not out:
        out.append(line)
    return out


def render_panel(lines: Sequence[str], width: int, height: int, fonts) -> np.ndarray:
    """상태 줄들 → 패널 이미지(BGR)."""
    from PIL import Image, ImageDraw

    title_font, font = fonts
    img = Image.new("RGB", (max(1, width), max(1, height)), _BG)
    draw = ImageDraw.Draw(img)
    x, y = 14, 10
    pad = 4
    for i, raw in enumerate(lines):
        is_title = i == 0
        accent = raw.startswith("# ")
        text = raw[2:] if accent else raw
        f = title_font if is_title else font
        color = _ACCENT if (is_title or accent) else _FG
        for part in _wrap(text, f, width - 2 * x):
            line_h = int(getattr(f, "size", 14)) + pad
            if y + line_h > height - 30:            # 아래는 배속·시간 자리
                break
            draw.text((x, y), part, font=f, fill=color)
            y += line_h
        if is_title:
            y += 4
            draw.line((x, y, width - x, y), fill=_DIM, width=1)
            y += 6
    arr = np.asarray(img, dtype=np.uint8)
    return arr[:, :, ::-1].copy()


__all__ = ["DemoRecorder", "render_panel", "vblank_address", "VBLANK_FALLBACK"]
