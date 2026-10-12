"""Parking session tracking — migrated from backend-parking-main."""

from __future__ import annotations

import threading
import time


class SpotEpisodeState:
    def __init__(self) -> None:
        self.occupied_prev: bool = False
        self.active_start_s: float | None = None
        self.total_sec: float = 0.0
        self.episodes_count: int = 0
        self.last_episode_duration_sec: float | None = None

    def update(self, occupied_now: bool, now_s: float) -> None:
        if occupied_now and not self.occupied_prev:
            self.active_start_s = now_s
            self.episodes_count += 1
        if not occupied_now and self.occupied_prev and self.active_start_s is not None:
            duration = max(0.0, now_s - self.active_start_s)
            self.total_sec += duration
            self.last_episode_duration_sec = duration
            self.active_start_s = None
        self.occupied_prev = occupied_now

    def current_sec(self, now_s: float) -> float:
        if self.active_start_s is None:
            return 0.0
        return max(0.0, now_s - self.active_start_s)


class ParkingTrackingSession:
    def __init__(self, spot_ids: list[str]) -> None:
        self.started_s = time.time()
        self.spot_states: dict[str, SpotEpisodeState] = {sid: SpotEpisodeState() for sid in spot_ids}
        self.paused: bool = False
        self.paused_start_s: float | None = None
        self.pause_snapshot_s: float | None = None
        self.lock = threading.Lock()

    def ensure_spots(self, spot_ids: list[str]) -> None:
        for sid in spot_ids:
            if sid not in self.spot_states:
                self.spot_states[sid] = SpotEpisodeState()

    def pause(self, now_s: float) -> None:
        with self.lock:
            if not self.paused:
                self.paused = True
                self.paused_start_s = now_s
                self.pause_snapshot_s = now_s

    def resume(self, now_s: float) -> None:
        with self.lock:
            if not self.paused:
                return
            dt = max(0.0, now_s - (self.paused_start_s or now_s))
            for state in self.spot_states.values():
                if state.active_start_s is not None:
                    state.active_start_s += dt
            self.paused = False
            self.paused_start_s = None
            self.pause_snapshot_s = None

    def update(self, occupied_by_spot: dict[str, bool], now_s: float) -> dict[str, dict]:
        with self.lock:
            self.ensure_spots(list(occupied_by_spot.keys()))
            if self.paused:
                effective_now = self.pause_snapshot_s or now_s
                meta = {
                    sid: {
                        "totalOccupiedSec": state.total_sec + state.current_sec(effective_now),
                        "currentOccupiedSec": state.current_sec(effective_now),
                        "episodesCount": state.episodes_count,
                        "lastEpisodeDurationSec": state.last_episode_duration_sec,
                    }
                    for sid, state in self.spot_states.items()
                }
                return {sid: meta[sid] for sid in occupied_by_spot if sid in meta}

            out: dict[str, dict] = {}
            for sid, occupied_now in occupied_by_spot.items():
                state = self.spot_states[sid]
                state.update(occupied_now=occupied_now, now_s=now_s)
                current_sec = state.current_sec(now_s)
                out[sid] = {
                    "totalOccupiedSec": state.total_sec + current_sec,
                    "currentOccupiedSec": current_sec,
                    "episodesCount": state.episodes_count,
                    "lastEpisodeDurationSec": state.last_episode_duration_sec,
                }
            return out

    def stop(self, now_s: float) -> None:
        with self.lock:
            for state in self.spot_states.values():
                if state.occupied_prev and state.active_start_s is not None:
                    duration = max(0.0, now_s - state.active_start_s)
                    state.total_sec += duration
                    state.last_episode_duration_sec = duration
                    state.active_start_s = None
                    state.occupied_prev = False
