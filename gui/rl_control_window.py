import os
import sys

# Ensure project root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gui.control_window import ControlWindow
from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv
from simulation.config import PROJECT_ROOT, MODELS_DIR, resolve_model_path

class RLControlWindow(ControlWindow):
    def __init__(self, master):
        self.model = None
        self.model_loaded = False
        self.active_model_path = ""
        self.current_step = 0
        super().__init__(master)

    def _configure_style(self):
        super()._configure_style()
        style = ttk.Style(self.master)
        style.configure(
            "TCombobox",
            fieldbackground="#0b1220",
            background="#1e293b",
            foreground="#e2e8f0",
            darkcolor="#1e293b",
            lightcolor="#1e293b",
            arrowcolor="#cbd5e1",
            padding=4,
        )
        self.master.option_add("*TCombobox*Listbox.background", "#0b1220")
        self.master.option_add("*TCombobox*Listbox.foreground", "#e2e8f0")
        self.master.option_add("*TCombobox*Listbox.selectBackground", "#334155")
        self.master.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        # Sleek modern notebook tab styles
        style.configure("TNotebook", background="#0f172a", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#1e293b",
            foreground="#94a3b8",
            padding=[16, 6],
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#334155"), ("active", "#253347")],
            foreground=[("selected", "#38bdf8"), ("active", "#ffffff")]
        )

    def _build_controls(self):
        super()._build_controls()

        # RL Model Selection Card
        model_card = tk.Frame(self.controls_container, bg="#111827", padx=10, pady=10)
        # Pack right above action card if desired, or at the bottom
        model_card.pack(fill="x", pady=(0, 10))

        ttk.Label(model_card, text="RL Model Selection", style="Section.TLabel").pack(anchor="w")
        ttk.Label(model_card, text="Select active policy checkpoint", style="Meta.TLabel").pack(anchor="w", pady=(0, 8))

        self.available_models = self._scan_available_models()
        default_choice = list(self.available_models.keys())[0] if self.available_models else ""
        self.selected_model_var = tk.StringVar(value=default_choice)

        selector_row = ttk.Frame(model_card, style="Card.TFrame")
        selector_row.pack(fill="x", pady=(0, 6))
        selector_row.columnconfigure(0, weight=1)

        self.model_combobox = ttk.Combobox(
            selector_row,
            textvariable=self.selected_model_var,
            values=list(self.available_models.keys()),
            state="readonly",
        )
        self.model_combobox.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_model_dropdown_changed)

        self.browse_model_button = ttk.Button(
            selector_row,
            text="Browse...",
            command=self._browse_model_file,
        )
        self.browse_model_button.grid(row=0, column=1, sticky="e")

        self.model_status_label = ttk.Label(
            model_card,
            text="Initializing model...",
            style="Meta.TLabel",
        )
        self.model_status_label.pack(anchor="w")

        # Load initial model
        if default_choice and default_choice in self.available_models:
            self.load_rl_model(self.available_models[default_choice])

    def _scan_available_models(self):
        models = {}
        # 1. Active default model from models/ directory
        default_path = resolve_model_path("td3_accel_agent.zip")
        if os.path.exists(default_path):
            models["Default (td3_accel_agent.zip)"] = os.path.abspath(default_path)

        # 2. Versioned models from models/ directory
        models_dir = str(MODELS_DIR)
        if os.path.exists(models_dir):
            for root, _, files in os.walk(models_dir):
                for f in sorted(files):
                    if f.endswith(".zip") and f != "td3_accel_agent.zip":
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, start=str(PROJECT_ROOT)).replace("\\", "/")
                        label = f"models/{f}" if root == models_dir else rel_path
                        models[label] = full_path

        return models

    def _on_model_dropdown_changed(self, _event=None):
        choice = self.selected_model_var.get()
        if choice in self.available_models:
            self.load_rl_model(self.available_models[choice])
        elif os.path.exists(choice):
            self.load_rl_model(choice)

    def _browse_model_file(self):
        filename = filedialog.askopenfilename(
            title="Select TD3 Model Checkpoint",
            filetypes=[("Zip Model Files", "*.zip"), ("All Files", "*.*")],
        )
        if filename:
            short_name = os.path.basename(filename)
            self.available_models[short_name] = filename
            self.model_combobox["values"] = list(self.available_models.keys())
            self.selected_model_var.set(short_name)
            self.load_rl_model(filename)

    def load_rl_model(self, model_path):
        if not os.path.isabs(model_path):
            abs_path = os.path.abspath(model_path)
        else:
            abs_path = model_path

        if os.path.exists(abs_path):
            try:
                self.model = TD3.load(abs_path)
                self.model_loaded = True
                self.active_model_path = abs_path
                base_name = os.path.basename(abs_path)
                print(f"Successfully loaded RL model from {abs_path}")
                if hasattr(self, "model_status_label"):
                    self.model_status_label.config(
                        text=f"✓ Active: {base_name}",
                        foreground="#4ade80",
                    )
            except Exception as e:
                self.model = None
                self.model_loaded = False
                print(f"Error loading RL model: {e}")
                if hasattr(self, "model_status_label"):
                    self.model_status_label.config(
                        text=f"✗ Error: {e}",
                        foreground="#f87171",
                    )
        else:
            self.model = None
            self.model_loaded = False
            print(f"Warning: RL model '{abs_path}' not found.")
            if hasattr(self, "model_status_label"):
                self.model_status_label.config(
                    text=f"⚠ Not found: {os.path.basename(abs_path)}",
                    foreground="#fbbf24",
                )

    def _build_visualization_area(self):
        self.right_content.columnconfigure(0, weight=1)

        # Metrics strip
        self.metrics_strip = tk.Frame(self.right_content, bg="#111827", padx=12, pady=8)
        self.metrics_strip.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.metrics_strip.columnconfigure((0, 1, 2), weight=1)

        self.metric_value_total = self._inline_metric(self.metrics_strip, 0, "Energy", "0.0000 kWh")
        self.metric_value_vel = self._inline_metric(self.metrics_strip, 1, "Avg Velocity", "0.00 m/s")
        self.metric_value_stability = self._inline_metric(self.metrics_strip, 2, "Stability", "0.0")

        # Create City objects
        self.city_acc = RLCity() # Standard ACC
        self.city_bcc = RLCity() # Standard BCC
        self.city_accbcc = RLCity() # Integrated ACC+BCC
        self.city_rl = RLCity()  # RL Controlled

        # Modern Notebook Tabs: Tab 1 = Simulation Road Tracks, Tab 2 = Live Real-Time Telemetry & Gaps
        self.notebook = ttk.Notebook(self.right_content)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        # Tab 1: Road Simulation View
        self.tab_simulation = ttk.Frame(self.notebook, style="Pane.TFrame")
        self.notebook.add(self.tab_simulation, text="  Road Simulation  ")

        self.visualizations_wrap = ttk.Frame(self.tab_simulation, style="Pane.TFrame")
        self.visualizations_wrap.pack(fill="both", expand=True)

        # Build visualization cards on Tab 1
        self.painter_acc, self.energy_label_acc, self.meta_acc = self._build_visualization_card("ACC", self.city_acc, parent=self.visualizations_wrap, height=140)
        self.painter_bcc, self.energy_label_bcc, self.meta_bcc = self._build_visualization_card("BCC", self.city_bcc, parent=self.visualizations_wrap, height=140)
        self.painter_accbcc, self.energy_label_accbcc, self.meta_accbcc = self._build_visualization_card("ACC+BCC Integration", self.city_accbcc, parent=self.visualizations_wrap, height=140)
        
        rl_title = "RL TD3 Agent (Direct Acceleration)"
        self.painter_rl, self.energy_label_rl, self.meta_rl = self._build_visualization_card(rl_title, self.city_rl, parent=self.visualizations_wrap, height=140)

        self.painters = [self.painter_acc, self.painter_bcc, self.painter_accbcc, self.painter_rl]

        # Tab 2: Live Real-Time Telemetry & Gap Profiles
        self.tab_telemetry = ttk.Frame(self.notebook, style="Pane.TFrame")
        self.notebook.add(self.tab_telemetry, text="  Real-Time Telemetry & Gaps  ")
        self._build_realtime_telemetry_tab()

    def _build_realtime_telemetry_tab(self):
        # Header banner in telemetry tab
        telemetry_header = tk.Frame(self.tab_telemetry, bg="#111827", padx=12, pady=8)
        telemetry_header.pack(fill="x", pady=(0, 4))
        
        title_box = tk.Frame(telemetry_header, bg="#111827")
        title_box.pack(side="left")
        
        tk.Label(
            title_box, 
            text="Live Real-Time Telemetry Stream", 
            fg="#f8fafc", 
            bg="#111827", 
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")
        
        tk.Label(
            title_box, 
            text="Real-time rendering of Velocity, Acceleration & Inter-Vehicle Gaps across all 4 controllers", 
            fg="#94a3b8", 
            bg="#111827", 
            font=("Segoe UI", 8)
        ).pack(anchor="w")

        btn_box = tk.Frame(telemetry_header, bg="#111827")
        btn_box.pack(side="right")
        
        self.live_badge = tk.Label(
            btn_box,
            text="LIVE STREAMING",
            fg="#22c55e",
            bg="#111827",
            font=("Consolas", 9, "bold"),
            padx=8
        )
        self.live_badge.pack(side="left", padx=(0, 10))

        ttk.Button(
            btn_box,
            text="Pop-out Window",
            command=self.plot_vel_acc_profiles
        ).pack(side="left")

        # Telemetry Matplotlib Figure (4 rows x 3 columns)
        self.telemetry_fig = Figure(figsize=(13.5, 9.8), dpi=95, facecolor="#0f172a")
        self.telemetry_fig.subplots_adjust(
            left=0.06, right=0.98, top=0.95, bottom=0.06, hspace=0.38, wspace=0.22
        )
        self.telemetry_axes = self.telemetry_fig.subplots(4, 3, sharex='col')

        self.telemetry_canvas = FigureCanvasTkAgg(self.telemetry_fig, master=self.tab_telemetry)
        canvas_widget = self.telemetry_canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        self._init_telemetry_artists()

    def _init_telemetry_artists(self):
        models = [
            ("ACC", self.city_acc),
            ("BCC", self.city_bcc),
            ("ACC+BCC Integration", self.city_accbcc),
            ("RL Direct Acceleration", self.city_rl)
        ]
        
        self.telemetry_lines = {}
        min_dis = getattr(self, 'min_dis', 6.0)

        for r, (name, city) in enumerate(models):
            ax_v = self.telemetry_axes[r, 0]
            ax_a = self.telemetry_axes[r, 1]
            ax_g = self.telemetry_axes[r, 2]

            for ax in (ax_v, ax_a, ax_g):
                ax.clear()
                ax.set_facecolor("#0b1220")
                ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.8)
                ax.tick_params(colors="#94a3b8", labelsize=7.5)
                for spine in ax.spines.values():
                    spine.set_color("#1e293b")

            ax_v.set_title(f"{name} Velocity", color="#e2e8f0", fontsize=8.5, fontweight="bold", pad=3)
            ax_v.set_ylabel("Vel (m/s)", color="#94a3b8", fontsize=8)
            ax_v.set_ylim(-1.0, 32.0)

            ax_a.set_title(f"{name} Acceleration", color="#e2e8f0", fontsize=8.5, fontweight="bold", pad=3)
            ax_a.set_ylabel("Accel (m/s²)", color="#94a3b8", fontsize=8)
            ax_a.set_ylim(-5.5, 3.5)

            ax_g.set_title(f"{name} Inter-Vehicle Gap", color="#e2e8f0", fontsize=8.5, fontweight="bold", pad=3)
            ax_g.set_ylabel("Gap (m)", color="#94a3b8", fontsize=8)
            ax_g.set_ylim(0.0, 45.0)

            # Dashed safety min_dis line
            ax_g.axhline(y=min_dis, color="#eab308", linestyle=":", linewidth=1.1, alpha=0.85)

            if r == 3:
                ax_v.set_xlabel("Time (s)", color="#cbd5e1", fontsize=8)
                ax_a.set_xlabel("Time (s)", color="#cbd5e1", fontsize=8)
                ax_g.set_xlabel("Time (s)", color="#cbd5e1", fontsize=8)

            lines_v, lines_a, lines_g = [], [], []
            cars = city.cars if city.cars else []
            for idx, car in enumerate(cars):
                color = "#ef4444" if idx == 0 else "#22c55e" if idx == 2 else "#64748b"
                lw = 1.4 if idx in (0, 2) else 0.75
                
                lv, = ax_v.plot([], [], color=color, linewidth=lw)
                la, = ax_a.plot([], [], color=color, linewidth=lw)
                lines_v.append(lv)
                lines_a.append(la)
                
                if idx > 0:
                    lg, = ax_g.plot([], [], color=color, linewidth=lw)
                    lines_g.append(lg)

            self.telemetry_lines[r] = {'vel': lines_v, 'acc': lines_a, 'gap': lines_g}

        self.telemetry_canvas.draw_idle()

    def _update_realtime_telemetry(self):
        if not hasattr(self, 'telemetry_lines') or not self.telemetry_lines:
            return
            
        dt = getattr(self, 'dt', 0.1)
        max_time = 5.0
        max_gap_seen = 35.0
        
        models = [
            (0, self.city_acc),
            (1, self.city_bcc),
            (2, self.city_accbcc),
            (3, self.city_rl)
        ]
        
        for r, city in models:
            lines_dict = self.telemetry_lines.get(r, None)
            if not lines_dict:
                continue
                
            for idx, car in enumerate(city.cars):
                n_v = len(car.vel_history)
                if n_v == 0:
                    continue
                time_axis = [dt * i for i in range(n_v)]
                if time_axis[-1] > max_time:
                    max_time = time_axis[-1]
                    
                if idx < len(lines_dict['vel']):
                    lines_dict['vel'][idx].set_data(time_axis, car.vel_history)
                if idx < len(lines_dict['acc']):
                    lines_dict['acc'][idx].set_data(time_axis, car.acc_history)
                if idx > 0 and (idx - 1) < len(lines_dict['gap']):
                    n_g = len(car.gap_history)
                    if n_g > 0:
                        t_gap = [dt * i for i in range(n_g)]
                        lines_dict['gap'][idx - 1].set_data(t_gap, car.gap_history)
                        cur_max_g = max(car.gap_history)
                        if cur_max_g > max_gap_seen:
                            max_gap_seen = cur_max_g

        # Smooth auto-scaling of time window
        xlim_right = max(5.0, max_time + 0.5)
        for ax in self.telemetry_axes.flat:
            ax.set_xlim(0, xlim_right)
            
        # If gap exceeded 45m, expand gap y-limits smoothly
        if max_gap_seen > 40.0:
            for r in range(4):
                self.telemetry_axes[r, 2].set_ylim(0, max_gap_seen + 5.0)

        self.telemetry_canvas.draw_idle()

    def _on_notebook_tab_changed(self, _event=None):
        try:
            if hasattr(self, 'notebook'):
                active_tab_idx = self.notebook.index(self.notebook.select())
                if active_tab_idx == 1:
                    self._update_realtime_telemetry()
        except Exception:
            pass

    def run_simulation(self):
        # Get parameter values from entry fields
        args = []
        param_keys = ["car_number", "kd", "kv", "kc", "v_des", "max_v", "min_v", "min_dis", "reaction_time","headway_time",  "max_a", "min_a", "min_gap", "dt"]
        for key in param_keys:
            val = self.entries[key].get()
            try:
                val = float(val) if '.' in val or 'e' in val.lower() else int(val)
            except Exception:
                val = 0
            args.append(val)

        self.dt = args[-1]  # Set self.dt from user input
        init_params = args[:-1]

        # Initialize cities
        self.city_acc.init(*init_params, dt=self.dt, model='ACC')
        self.city_bcc.init(*init_params, dt=self.dt, model='BCC')
        self.city_accbcc.init(*init_params, dt=self.dt, model='ACC+BCC')
        self.city_rl.init(*init_params, dt=self.dt, model='ACC')

        # Update painters with new city elements
        self.painter_acc.set_elements(self.city_acc.roads, self.city_acc.cars)
        self.painter_bcc.set_elements(self.city_bcc.roads, self.city_bcc.cars)
        self.painter_accbcc.set_elements(self.city_accbcc.roads, self.city_accbcc.cars)
        self.painter_rl.set_elements(self.city_rl.roads, self.city_rl.cars)

        # Refresh real-time telemetry artists with new car fleet
        if hasattr(self, 'telemetry_canvas'):
            self._init_telemetry_artists()

        # Set cars to RL mode (except the lead car)
        for i, car in enumerate(self.city_rl.cars):
            if i > 0:
                car.is_rl = True
                car.rl_action = 0.0
            else:
                car.is_rl = False
                car.mode = 'ACC'

        # Handle velocity profiles if enabled
        if self.use_velocity_profile.get():
            self.load_velocity_profile()
            for city in [self.city_acc, self.city_bcc, self.city_accbcc, self.city_rl]:
                city.lead_velocity_profile = self.ego_velocity_profile
                city.follower_velocity_profile = self.ego_velocity_profile_1
        else:
            for city in [self.city_acc, self.city_bcc, self.city_accbcc, self.city_rl]:
                city.lead_velocity_profile = []
                city.follower_velocity_profile = []

        self.current_step = 0
        self.start_timer()

    def update_simulation(self):
        dt = self.dt
        self.current_step += 1
        
        # Handle standard logic for ACC and BCC
        self.city_acc.set_leader_stop(self.leader_stop)
        self.city_acc.set_follower_stop(self.follower_stop)
        self.city_bcc.set_leader_stop(self.leader_stop)
        self.city_bcc.set_follower_stop(self.follower_stop)
        self.city_accbcc.set_leader_stop(self.leader_stop)
        self.city_accbcc.set_follower_stop(self.follower_stop)
        
        self.city_acc.run(dt)
        self.city_bcc.run(dt)
        self.city_accbcc.run(dt)

        # Handle RL Agent Logic
        if self.model_loaded and self.city_rl.cars:
            self.city_rl.set_leader_stop(self.leader_stop)
            self.city_rl.set_follower_stop(self.follower_stop)
            
            try:
                # 1. Get Observations for RL cars
                obs = self._get_rl_observation(self.city_rl)
                
                # 2. Predict Acceleration
                # TD3 predicts a vector of accelerations for all cars
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 3. Apply raw actions directly to FOLLOWER cars (indices 1..N-1)
                # No EMA smoothing — rl_city.py's jerk limiter handles smoothness
                for i, car in enumerate(self.city_rl.cars[1:]):
                    if i < len(action):
                        car.rl_action = float(action[i][0])
            except Exception as e:
                if self.current_step <= 1: 
                    print(f"RL Prediction/Observation error: {e}")
            
            # 4. Run RL City Step
            self.city_rl.run(dt)

        # Redraw all painters
        for p in self.painters:
            p.repaint()

        # Update metrics and labels
        total_energy_acc = sum(car.energy_used for car in self.city_acc.cars)
        total_energy_bcc = sum(car.energy_used for car in self.city_bcc.cars)
        total_energy_accbcc = sum(car.energy_used for car in self.city_accbcc.cars)
        total_energy_rl = sum(car.energy_used for car in self.city_rl.cars)

        self.energy_label_acc.config(text=f"Energy {total_energy_acc:.4f} kWh")
        self.energy_label_bcc.config(text=f"Energy {total_energy_bcc:.4f} kWh")
        self.energy_label_accbcc.config(text=f"Energy {total_energy_accbcc:.4f} kWh")
        self.energy_label_rl.config(text=f"Energy {total_energy_rl:.4f} kWh")

        # Avg velocities
        def get_avg_v(city):
            return sum(car.velocity for car in city.cars) / len(city.cars) if city.cars else 0.0

        self.meta_acc.config(text=f"Cars {len(self.city_acc.cars):02d} | Avg v {get_avg_v(self.city_acc):05.2f} m/s")
        self.meta_bcc.config(text=f"Cars {len(self.city_bcc.cars):02d} | Avg v {get_avg_v(self.city_bcc):05.2f} m/s")
        self.meta_accbcc.config(text=f"Cars {len(self.city_accbcc.cars):02d} | Avg v {get_avg_v(self.city_accbcc):05.2f} m/s")
        self.meta_rl.config(text=f"Cars {len(self.city_rl.cars):02d} | Avg v {get_avg_v(self.city_rl):05.2f} m/s")

        # Global metrics strip update
        total_energy_all = total_energy_acc + total_energy_bcc + total_energy_accbcc + total_energy_rl
        self.metric_value_total.config(text=f"{total_energy_all:08.4f} kWh")
        self.metric_value_vel.config(text=f"{get_avg_v(self.city_rl):05.2f} m/s") # Show RL speed as primary

        # Capture video frame if recording is active
        if hasattr(self, "recorder") and self.recorder.is_recording:
            target = getattr(self, "root", self.master)
            self.recorder.capture_frame(target)
            if self.recorder.frame_count % 10 == 0:
                self.record_status_label.config(text=f"🔴 Rec: {self.recorder.frame_count} frames")

        # Stream live real-time telemetry if user is viewing the telemetry tab (every 2 steps = 5 FPS)
        if hasattr(self, "notebook"):
            try:
                active_tab_idx = self.notebook.index(self.notebook.select())
                if active_tab_idx == 1 and (self.current_step % 2 == 0):
                    self._update_realtime_telemetry()
            except Exception:
                pass

        # Schedule next update
        self.timer = self.master.after(int(dt*1000), self.update_simulation)

    def _get_rl_observation(self, city):
        v_des = getattr(city, 'v_des', 15.0)
        return AccelTrafficEnv.get_city_obs(city, v_des)

    def plot_vel_acc_profiles(self):
        dt = self.dt

        fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex='col')

        models = [
            ("ACC", self.city_acc),
            ("BCC", self.city_bcc),
            ("ACC+BCC Integration", self.city_accbcc),
            ("RL Direct Acceleration", self.city_rl)
        ]

        min_dis = getattr(self, 'min_dis', 6.0)

        for row, (name, city) in enumerate(models):
            # 1. Velocity (Col 0)
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.vel_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx in (0, 2) else 0.5
                axes[row, 0].plot(time_axis, car.vel_history, color=color, linewidth=linewidth)
            
            axes[row, 0].set_title(f"{name} Velocity")
            axes[row, 0].set_ylabel("Velocity (m/s)")
            axes[row, 0].grid(True)
            if row == 3: axes[row, 0].set_xlabel("Time (s)")

            # 2. Acceleration (Col 1)
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.acc_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx in (0, 2) else 0.5
                axes[row, 1].plot(time_axis, car.acc_history, color=color, linewidth=linewidth)
                
            axes[row, 1].set_title(f"{name} Acceleration")
            axes[row, 1].set_ylabel("Accel (m/s^2)")
            axes[row, 1].grid(True)
            if row == 3: axes[row, 1].set_xlabel("Time (s)")

            # 3. Inter-Vehicle Gap (Col 2)
            for idx, car in enumerate(city.cars):
                if idx == 0:
                    continue
                time_axis = [dt * i for i in range(len(car.gap_history))]
                color = "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx == 2 else 0.5
                axes[row, 2].plot(time_axis, car.gap_history, color=color, linewidth=linewidth)
                
            axes[row, 2].axhline(y=min_dis, color="orange", linestyle="--", linewidth=1.0, alpha=0.8, label=f"min_dis ({min_dis}m)")
            axes[row, 2].set_title(f"{name} Inter-Vehicle Gap")
            axes[row, 2].set_ylabel("Gap (m)")
            axes[row, 2].grid(True)
            if row == 3: axes[row, 2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("RL-Enhanced Traffic Simulation")
    app = RLControlWindow(root)
    root.mainloop()
