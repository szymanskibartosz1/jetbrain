"""
Dataset Generator

assumptions:
    - precursor signal is weak and buried in noise
    - some spikes in metrics are just noise, not incidents
    - lead time before incident varies (5 to 60 steps)
    - latency only reacts when cpu crosses ~50% (non-linear)
    - baseline shifts over time as traffic patterns change
    - 20% of incidents hit with no warning at all
    - system operates in different load regimes throughout the day
"""

# imports
import numpy as np
import pandas as pd


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) process"""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sigma)

    return x


def generate_dataset(
    n_samples: int = 15_000,
    n_incidents: int = 50,
    seed: int = 132,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    t = np.arange(n_samples, dtype=float)

    # baseline shifts over time - system goes through low/mid/high traffic phases
    regime = (
        0.3 * np.tanh((t - n_samples * 0.3) / 800)
        + 0.3 * np.tanh((t - n_samples * 0.65) / 600)
    )  # in [-0.6, 0.6]

    # two overlapping cycles — daily (1440 min) and weekly (10080 min)
    diurnal  = np.sin(2 * np.pi * t / 1440)
    weekly   = 0.4 * np.sin(2 * np.pi * t / 10080)
    seasonal = diurnal + weekly

    # AR(1) correlated noise per metric
    noise_cpu    = _ar1(n_samples, phi=0.7, sigma=2.5, rng=rng)
    noise_mem    = _ar1(n_samples, phi=0.85, sigma=1.5, rng=rng)
    noise_lat    = _ar1(n_samples, phi=0.5, sigma=3.0, rng=rng)
    noise_errrate = _ar1(n_samples, phi=0.4, sigma=0.3, rng=rng)

    # cpu hovers around 40%, climbs during busy periods
    cpu = (
        42 + 8 * regime
        + 6 * seasonal
        + noise_cpu
        + rng.normal(0, 1.5, n_samples)
    )

    # memory creeps up slowly, drops when GC kicks in
    mem_trend = 58 + 0.001 * t + 5 * regime
    gc_drops  = np.zeros(n_samples)
    gc_times  = rng.integers(0, n_samples, size=80)

    for gc in gc_times:
        end = min(gc + 5, n_samples)
        gc_drops[gc:end] -= rng.uniform(3, 8)

    memory = mem_trend + 4 * seasonal + noise_mem + gc_drops + rng.normal(0, 1.2, n_samples)
    memory = np.clip(memory, 10, 95)

    # latency is log-normal and only reacts when cpu gets high
    cpu_load_effect = np.maximum(0, cpu - 50) * 0.8   # latency rises when CPU > 50%
    log_lat = (
        4.6
        + 0.02 * cpu_load_effect
        + 0.3 * seasonal
        + 0.002 * noise_lat
        + rng.normal(0, 0.12, n_samples)
    )

    latency = np.exp(log_lat)

    # Error ratee
    error_rate = np.clip(
        0.5
        + 0.01 * np.maximum(0, latency - 200)
        + noise_errrate
        + rng.exponential(0.1, n_samples),
        0, 100
    )

    incident_label = np.zeros(n_samples, dtype=int)

    # 3. Incidents                                                   
    incident_duration = rng.integers(15, 40, size=n_incidents)

    # Spread incident starts across full timeline
    segment = (n_samples - 100) // n_incidents
    incident_starts = []


    for k in range(n_incidents):
        seg_s = 60 + k * segment
        seg_e = seg_s + segment - 50

        if seg_e <= seg_s:
            seg_e = seg_s + 30
        start = int(rng.integers(seg_s, seg_e))
        incident_starts.append(start)

    for idx, (start, dur) in enumerate(zip(incident_starts, incident_duration)):
        end = min(start + int(dur), n_samples)
        incident_label[start:end] = 1

        has_precursor = rng.random() > 0.20    # 1 in 5 incidents hit with no warning
        lead_time     = int(rng.integers(5, 61))  # how far before the incident the signal starts
        pre_start     = max(0, start - lead_time)
        ramp_len      = start - pre_start

        if has_precursor and ramp_len > 0:
            # signal builds slowly at first, then accelerates toward the incident
            ramp = np.linspace(0, 1, ramp_len) ** 1.5  
            noise_scale = rng.uniform(0.4, 0.9)  

            cpu_boost    = rng.uniform(8, 20)
            mem_boost    = rng.uniform(5, 15)
            lat_mult     = rng.uniform(0.3, 1.2)
            err_boost    = rng.uniform(0.5, 3.0)

            cpu[pre_start:start]    += ramp * cpu_boost    + rng.normal(0, noise_scale * 3, ramp_len)
            memory[pre_start:start] += ramp * mem_boost    + rng.normal(0, noise_scale * 2, ramp_len)
            latency[pre_start:start] *= (1 + ramp * lat_mult + rng.normal(0, noise_scale * 0.1, ramp_len))
            error_rate[pre_start:start] += ramp * err_boost

        # metrics spike hard during the incident itself
        severity = rng.uniform(0.6, 1.0)
        cpu[start:end]    = np.clip(cpu[start:end] + rng.uniform(25, 50, end-start) * severity, 0, 100)
        memory[start:end] = np.clip(memory[start:end] + rng.uniform(15, 35, end-start) * severity, 0, 100)
        latency[start:end] *= rng.uniform(2, 8, end-start) * severity
        error_rate[start:end] += rng.uniform(5, 30, end-start) * severity

    # throw in some random spikes that look alarming but aren't incidents
    n_false_alarms = 80 
    fa_times = rng.integers(60, n_samples - 20, size=n_false_alarms)

    for fa in fa_times:
        # skip if too close to a real incident
        window = incident_label[max(0, fa-30):min(n_samples, fa+30)]
        if window.sum() > 0:
            continue

        fa_dur = int(rng.integers(3, 12))
        end_fa = min(fa + fa_dur, n_samples)


        which = rng.integers(0, 3)

        if which == 0:
            cpu[fa:end_fa]    += rng.uniform(15, 35, end_fa - fa)
        elif which == 1:
            latency[fa:end_fa] *= rng.uniform(2, 4, end_fa - fa)
        else:
            memory[fa:end_fa] += rng.uniform(10, 25, end_fa - fa)

   
    cpu        = np.clip(cpu, 0, 100)
    memory     = np.clip(memory, 0, 100)
    latency    = np.clip(latency, 50, 15000)
    error_rate = np.clip(error_rate, 0, 100)

    df = pd.DataFrame({
        "timestamp":     pd.date_range("2026-01-09", periods=n_samples, freq="1min"),
        "cpu":           cpu,
        "memory":        memory,
        "latency":       latency,
        "error_rate":    error_rate,
        "incident_label": incident_label,})

    return df

    
if __name__ == "__main__":
    df = generate_dataset()
    print(df.describe())
    print(f"\nincidents: {df['incident_label'].sum()} steps ({df['incident_label'].mean():.2%})")
