"""
Progress tracking with cancellation support for THOTH.

This module provides thread-safe progress tracking for long-running
translation operations, with support for cancellation and ETA estimation.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Queue
from typing import Callable, Optional


@dataclass
class ProgressState:
    """Current state of a translation operation."""

    # Total items to process
    total: int = 0

    # Items processed so far
    current: int = 0

    # Current operation description
    message: str = ""

    # Start time of the operation
    start_time: Optional[float] = None

    # Whether the operation has been cancelled
    cancelled: bool = False

    # Whether the operation is complete
    complete: bool = False

    # Error message if operation failed
    error: Optional[str] = None

    @property
    def percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def elapsed_formatted(self) -> str:
        """Get elapsed time as formatted string (HH:MM:SS)."""
        elapsed = int(self.elapsed_seconds)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimate seconds remaining."""
        if self.current == 0 or self.start_time is None:
            return None

        elapsed = self.elapsed_seconds
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = self.total - self.current

        if rate > 0:
            return remaining / rate
        return None

    @property
    def eta_formatted(self) -> str:
        """Get ETA as formatted string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."

        eta_int = int(eta)
        hours, remainder = divmod(eta_int, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @property
    def items_per_second(self) -> float:
        """Get processing rate in items per second."""
        elapsed = self.elapsed_seconds
        if elapsed > 0:
            return self.current / elapsed
        return 0.0


class ProgressTracker:
    """
    Thread-safe progress tracker with cancellation support.

    Tracks progress of long-running operations and provides callbacks
    for progress updates. Supports cancellation from any thread.

    Example:
        tracker = ProgressTracker(total=1000)

        # In worker thread
        for i in range(1000):
            if tracker.is_cancelled():
                break
            # Do work...
            tracker.update(1, f"Processing item {i}")

        tracker.complete()

        # In UI thread
        tracker.on_progress = lambda state: update_ui(state)
        tracker.cancel()  # Request cancellation
    """

    def __init__(
        self,
        total: int = 0,
        message: str = "",
        on_progress: Optional[Callable[[ProgressState], None]] = None,
    ) -> None:
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            message: Initial status message
            on_progress: Callback function for progress updates
        """
        self._state = ProgressState(total=total, message=message)
        self._lock = threading.Lock()
        self._on_progress = on_progress
        self._update_queue: Queue[ProgressState] = Queue()

    @property
    def state(self) -> ProgressState:
        """Get current progress state (thread-safe copy)."""
        with self._lock:
            return ProgressState(
                total=self._state.total,
                current=self._state.current,
                message=self._state.message,
                start_time=self._state.start_time,
                cancelled=self._state.cancelled,
                complete=self._state.complete,
                error=self._state.error,
            )

    @property
    def on_progress(self) -> Optional[Callable[[ProgressState], None]]:
        """Get progress callback."""
        return self._on_progress

    @on_progress.setter
    def on_progress(self, callback: Optional[Callable[[ProgressState], None]]) -> None:
        """Set progress callback."""
        self._on_progress = callback

    def start(self, total: Optional[int] = None, message: str = "") -> None:
        """
        Start tracking progress.

        Args:
            total: Optional new total (uses existing if not provided)
            message: Initial status message
        """
        with self._lock:
            if total is not None:
                self._state.total = total
            self._state.current = 0
            self._state.message = message
            self._state.start_time = time.time()
            self._state.cancelled = False
            self._state.complete = False
            self._state.error = None

        self._notify()

    def update(self, increment: int = 1, message: Optional[str] = None) -> None:
        """
        Update progress.

        Args:
            increment: Number of items completed since last update
            message: Optional new status message
        """
        with self._lock:
            self._state.current += increment
            if message is not None:
                self._state.message = message

        self._notify()

    def set_progress(self, current: int, message: Optional[str] = None) -> None:
        """
        Set absolute progress value.

        Args:
            current: Current progress value
            message: Optional new status message
        """
        with self._lock:
            self._state.current = current
            if message is not None:
                self._state.message = message

        self._notify()

    def set_message(self, message: str) -> None:
        """
        Update status message without changing progress.

        Args:
            message: New status message
        """
        with self._lock:
            self._state.message = message

        self._notify()

    def cancel(self) -> None:
        """Request cancellation of the operation."""
        with self._lock:
            self._state.cancelled = True

        self._notify()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        with self._lock:
            return self._state.cancelled

    def complete(self, message: str = "Complete") -> None:
        """
        Mark operation as complete.

        Args:
            message: Completion status message
        """
        with self._lock:
            self._state.complete = True
            self._state.message = message
            self._state.current = self._state.total

        self._notify()

    def fail(self, error: str) -> None:
        """
        Mark operation as failed.

        Args:
            error: Error message describing the failure
        """
        with self._lock:
            self._state.error = error
            self._state.complete = True

        self._notify()

    def reset(self) -> None:
        """Reset tracker to initial state."""
        with self._lock:
            self._state = ProgressState()

        self._notify()

    def _notify(self) -> None:
        """Notify callback of progress update."""
        if self._on_progress:
            state = self.state
            try:
                self._on_progress(state)
            except Exception:
                # Don't let callback errors break the tracker
                pass

        # Also queue for polling
        self._update_queue.put(self.state)

    def get_update(self, timeout: Optional[float] = None) -> Optional[ProgressState]:
        """
        Get next progress update from queue (for polling).

        Args:
            timeout: Maximum time to wait (None = non-blocking)

        Returns:
            ProgressState if available, None otherwise
        """
        try:
            if timeout is None:
                return self._update_queue.get_nowait()
            return self._update_queue.get(timeout=timeout)
        except Exception:
            return None

    def drain_updates(self) -> list[ProgressState]:
        """
        Get all pending updates from queue.

        Returns:
            List of all pending ProgressState updates
        """
        updates = []
        while True:
            try:
                updates.append(self._update_queue.get_nowait())
            except Exception:
                break
        return updates


class BatchProgressTracker(ProgressTracker):
    """
    Progress tracker optimized for batch operations.

    Provides additional methods for tracking batch-level progress
    and automatically calculates batch statistics.
    """

    def __init__(
        self,
        total_items: int,
        batch_size: int,
        on_progress: Optional[Callable[[ProgressState], None]] = None,
    ) -> None:
        """
        Initialize batch progress tracker.

        Args:
            total_items: Total number of items to process
            batch_size: Number of items per batch
            on_progress: Callback function for progress updates
        """
        super().__init__(total=total_items, on_progress=on_progress)
        self.batch_size = batch_size
        self._current_batch = 0
        self._total_batches = (total_items + batch_size - 1) // batch_size

    @property
    def total_batches(self) -> int:
        """Get total number of batches."""
        return self._total_batches

    @property
    def current_batch(self) -> int:
        """Get current batch number (1-indexed)."""
        return self._current_batch

    def start_batch(self, batch_num: int) -> None:
        """
        Signal start of a new batch.

        Args:
            batch_num: Batch number (1-indexed)
        """
        self._current_batch = batch_num
        self.set_message(f"Processing batch {batch_num}/{self._total_batches}")

    def complete_batch(self, items_processed: int) -> None:
        """
        Signal completion of current batch.

        Args:
            items_processed: Number of items completed in this batch
        """
        self.update(items_processed)


class MultiStageProgressTracker:
    """
    Progress tracker for multi-stage operations.

    Tracks progress across multiple stages (e.g., detect -> translate -> save)
    and provides overall progress calculation.
    """

    @dataclass
    class Stage:
        """Definition of a processing stage."""

        name: str
        weight: float = 1.0
        tracker: Optional[ProgressTracker] = None

    def __init__(
        self,
        stages: list[tuple[str, float]],
        on_progress: Optional[Callable[[ProgressState], None]] = None,
    ) -> None:
        """
        Initialize multi-stage tracker.

        Args:
            stages: List of (name, weight) tuples defining stages
            on_progress: Callback for overall progress updates
        """
        self._stages = [
            self.Stage(name=name, weight=weight)
            for name, weight in stages
        ]
        self._current_stage_idx = 0
        self._on_progress = on_progress
        self._cancelled = False

        # Normalize weights
        total_weight = sum(s.weight for s in self._stages)
        for stage in self._stages:
            stage.weight /= total_weight

    @property
    def current_stage(self) -> Optional[Stage]:
        """Get current stage."""
        if 0 <= self._current_stage_idx < len(self._stages):
            return self._stages[self._current_stage_idx]
        return None

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress percentage (0-100)."""
        progress = 0.0

        for i, stage in enumerate(self._stages):
            if i < self._current_stage_idx:
                # Completed stages
                progress += stage.weight * 100
            elif i == self._current_stage_idx and stage.tracker:
                # Current stage
                progress += stage.weight * stage.tracker.state.percentage

        return progress

    def start_stage(self, stage_name: str, total: int) -> ProgressTracker:
        """
        Start a new stage and return its tracker.

        Args:
            stage_name: Name of the stage to start
            total: Total items in this stage

        Returns:
            ProgressTracker for this stage
        """
        # Find stage by name
        for i, stage in enumerate(self._stages):
            if stage.name == stage_name:
                self._current_stage_idx = i
                stage.tracker = ProgressTracker(
                    total=total,
                    message=f"Starting {stage_name}...",
                    on_progress=self._stage_progress_callback,
                )
                stage.tracker.start()
                return stage.tracker

        raise ValueError(f"Unknown stage: {stage_name}")

    def _stage_progress_callback(self, state: ProgressState) -> None:
        """Internal callback to update overall progress."""
        if self._on_progress:
            overall_state = ProgressState(
                total=100,
                current=int(self.overall_progress),
                message=state.message,
                cancelled=self._cancelled or state.cancelled,
                complete=all(
                    s.tracker and s.tracker.state.complete
                    for s in self._stages
                ),
            )
            self._on_progress(overall_state)

    def cancel(self) -> None:
        """Cancel all stages."""
        self._cancelled = True
        for stage in self._stages:
            if stage.tracker:
                stage.tracker.cancel()

    def is_cancelled(self) -> bool:
        """Check if cancellation requested."""
        return self._cancelled


def format_progress_bar(
    percentage: float,
    width: int = 40,
    filled_char: str = "█",
    empty_char: str = "░",
) -> str:
    """
    Create a text-based progress bar.

    Args:
        percentage: Progress percentage (0-100)
        width: Width of the progress bar in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        Formatted progress bar string
    """
    filled = int(width * percentage / 100)
    empty = width - filled
    return f"{filled_char * filled}{empty_char * empty}"


def format_progress_line(state: ProgressState, bar_width: int = 30) -> str:
    """
    Create a complete progress line for CLI output.

    Args:
        state: Current progress state
        bar_width: Width of the progress bar

    Returns:
        Formatted progress line
    """
    bar = format_progress_bar(state.percentage, width=bar_width)
    pct = f"{state.percentage:5.1f}%"
    count = f"{state.current:,}/{state.total:,}"
    eta = f"ETA: {state.eta_formatted}"

    return f"{bar} {pct} | {count} | {eta}"
