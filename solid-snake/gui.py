from __future__ import annotations

import dataclasses
import tkinter as tk
from tkinter import filedialog, messagebox
import typing as t


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """
    DESCRIPTION: Configuration values collected from the GUI.

    PARAMETERS: show (REQ, bool) - If True, render the pygame window.
                train (REQ, bool) - If True, train the agent while playing.
                load (REQ, bool) - If True, attempt to load a checkpoint on startup.
                episodes (REQ, int) - Number of episodes to run.
                checkpoint_path (REQ, str) - Model checkpoint file path.
                difficulty_fps (REQ, int) - FPS limit for rendering (pygame clock tick).
                buffer_steps (REQ, int) - Loop-break buffer steps (passed to env.step).

    RETURNS: RunConfig - The run configuration.
    """
    show: bool
    train: bool
    load: bool
    episodes: int
    checkpoint_path: str
    difficulty_fps: int
    buffer_steps: int


class App(tk.Tk):
    def __init__(self) -> None:
        """
        DESCRIPTION: Tkinter GUI for configuring and launching the Snake DQN run.

        PARAMETERS: None

        RETURNS: None
        """
        super().__init__()
        self.title("Snake DQN - Run Config")
        self.resizable(False, False)

        # Defaults (safe + demo-friendly)
        self.var_show = tk.BooleanVar(value=True)
        self.var_train = tk.BooleanVar(value=False)
        self.var_load = tk.BooleanVar(value=True)

        self.var_episodes = tk.StringVar(value="500")
        self.var_checkpoint = tk.StringVar(value="snake_dqn.pt")
        self.var_difficulty = tk.StringVar(value="60")
        self.var_buffer_steps = tk.StringVar(value="1000")

        self._result: RunConfig | None = None

        self._build()

    def _build(self) -> None:
        """
        DESCRIPTION: Build and layout GUI widgets.

        PARAMETERS: None

        RETURNS: None
        """
        pad = {"padx": 10, "pady": 6}

        frm = tk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        # Flags
        flags = tk.LabelFrame(frm, text="Options")
        flags.grid(row=0, column=0, columnspan=3, sticky="ew", **pad)

        tk.Checkbutton(flags, text="Render (show pygame window)", variable=self.var_show).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        tk.Checkbutton(flags, text="Train (learn during play)", variable=self.var_train).grid(row=1, column=0, sticky="w", padx=8, pady=4)
        tk.Checkbutton(flags, text="Load checkpoint on start", variable=self.var_load).grid(row=2, column=0, sticky="w", padx=8, pady=4)

        # Values
        vals = tk.LabelFrame(frm, text="Parameters")
        vals.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)

        tk.Label(vals, text="Episodes").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        tk.Entry(vals, textvariable=self.var_episodes, width=12).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        tk.Label(vals, text="Render FPS (difficulty)").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        tk.Entry(vals, textvariable=self.var_difficulty, width=12).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        tk.Label(vals, text="Loop-break buffer steps").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        tk.Entry(vals, textvariable=self.var_buffer_steps, width=12).grid(row=2, column=1, sticky="w", padx=8, pady=4)

        tk.Label(vals, text="Checkpoint file").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        tk.Entry(vals, textvariable=self.var_checkpoint, width=32).grid(row=3, column=1, sticky="w", padx=8, pady=4)

        tk.Button(vals, text="Browse...", command=self._browse_checkpoint).grid(row=3, column=2, sticky="w", padx=8, pady=4)

        # Buttons
        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=3, sticky="e", **pad)

        tk.Button(btns, text="Run", width=12, command=self._on_run).pack(side="right", padx=6)
        tk.Button(btns, text="Cancel", width=12, command=self._on_cancel).pack(side="right", padx=6)

        # Helper text
        help_txt = (
            "Tips:\n"
            "• For fast training: uncheck Render.\n"
            "• For a demo: check Load, uncheck Train.\n"
            "• If Load fails (no file), it will start fresh."
        )
        tk.Label(frm, text=help_txt, justify="left").grid(row=3, column=0, columnspan=3, sticky="w", **pad)

    def _browse_checkpoint(self) -> None:
        """
        DESCRIPTION: Open a file dialog to choose a checkpoint file.

        PARAMETERS: None

        RETURNS: None
        """
        filename = filedialog.asksaveasfilename(
            title="Choose checkpoint file",
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialfile=self.var_checkpoint.get() or "snake_dqn.pt",
        )
        if filename:
            self.var_checkpoint.set(filename)

    def _parse_int(self, value: str, field: str, min_value: int = 1, max_value: int = 10_000_000) -> int:
        """
        DESCRIPTION: Parse an integer field with validation bounds.

        PARAMETERS: value (REQ, str) - String value from entry.
                    field (REQ, str) - Field name for error messages.
                    min_value (OPT, int), by default 1 - Minimum allowed.
                    max_value (OPT, int), by default 10_000_000 - Maximum allowed.

        RETURNS: int - Parsed integer.

        EXCEPTIONS: ValueError - If parsing fails or value is out of bounds.
        """
        v = int(value.strip())
        if v < min_value or v > max_value:
            raise ValueError(f"{field} must be between {min_value} and {max_value}")
        return v

    def _on_run(self) -> None:
        """
        DESCRIPTION: Validate inputs, store RunConfig, and close window.

        PARAMETERS: None

        RETURNS: None
        """
        try:
            episodes = self._parse_int(self.var_episodes.get(), "Episodes", min_value=1, max_value=1_000_000)
            difficulty = self._parse_int(self.var_difficulty.get(), "Render FPS", min_value=1, max_value=1000)
            buffer_steps = self._parse_int(self.var_buffer_steps.get(), "Buffer steps", min_value=50, max_value=10_000_000)

            checkpoint = self.var_checkpoint.get().strip()
            if not checkpoint:
                raise ValueError("Checkpoint file cannot be empty")

            self._result = RunConfig(
                show=bool(self.var_show.get()),
                train=bool(self.var_train.get()),
                load=bool(self.var_load.get()),
                episodes=episodes,
                checkpoint_path=checkpoint,
                difficulty_fps=difficulty,
                buffer_steps=buffer_steps,
            )
            self.destroy()

        except Exception as e:
            messagebox.showerror("Invalid configuration", str(e))

    def _on_cancel(self) -> None:
        """
        DESCRIPTION: Cancel and close without returning a config.

        PARAMETERS: None

        RETURNS: None
        """
        self._result = None
        self.destroy()

    def get_result(self) -> RunConfig | None:
        """
        DESCRIPTION: Return the RunConfig selected by the user, if any.

        PARAMETERS: None

        RETURNS: RunConfig | None - Selected configuration, or None if cancelled.
        """
        return self._result


def get_run_config() -> RunConfig | None:
    """
    DESCRIPTION: Launch the GUI and return the chosen RunConfig.

    PARAMETERS: None

    RETURNS: RunConfig | None - Run configuration or None if cancelled.
    """
    app = App()
    app.mainloop()
    return app.get_result()
