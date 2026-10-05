import model_selfschedule
import numpy as np
import pandas as pd
import contextlib
import io
import data_read
import os
import sys
from datetime import datetime


class Tee:
    """Write console output to a log file simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_test(data, storage_id, day, schedules=None, output_dir="output", appointed_pdis=None, appointed_pch=None, appointed_e=None):
    storage = data.storages[storage_id - 1]
    T = data.T
    signal = data.signal
    selected_signal = signal[day - 1, :]
    price_ch = None
    price_dis = None
    solver = "copt" 
    model_type = "Basic"
    obj_type = "SetPointTracking"  # Run the STP test
    run_milp, run_ch, run_lp = 1, 1, 1

    def build_schedule(**kwargs):
        kwargs["config"]["silent"] = True
        kwargs["config"]["save_results"] = False
        kwargs["config"]["output_dir"] = output_dir
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return model_selfschedule.SelfScheduleStorage(**kwargs)

    if schedules is None:
        schedules = {}

    if not schedules:
        if run_milp:
            schedules["MILP"] = build_schedule(
                storage = storage, 
                price_ch = price_ch if obj_type == "Profit" else None, 
                price_dis =  price_dis if obj_type == "Profit" else None,
                signal = selected_signal if obj_type == "SetPointTracking" else None, 
                T = T, 
                model_name=f"MILP_{model_type}_{obj_type}_storage_{storage_id}", 
                config={"solve_method": "MILP", "model_type": model_type, "obj_type": obj_type, "solver": solver}
            )

        if run_ch:
            schedules["Network CH"] = build_schedule(
                storage = storage,
                price_ch = price_ch if obj_type == "Profit" else None,
                price_dis = price_dis if obj_type == "Profit" else None,
                signal = selected_signal if obj_type == "SetPointTracking" else None,
                T = T,
                model_name=f"CH_{model_type}_{obj_type}_Network_storage_{storage_id}",
                config={"solve_method": "CH", "model_type": model_type, "obj_type": obj_type, "ch_method": "Network", "solver": solver}
            )
            if model_type == "Basic":
                for ch_method in ["VCH", "HCH"]:
                # for ch_method in ["HCH"]:
                    schedules[ch_method] = build_schedule(
                        storage = storage,
                        price_ch = price_ch if obj_type == "Profit" else None,
                        price_dis = price_dis if obj_type == "Profit" else None,
                        signal = selected_signal if obj_type == "SetPointTracking" else None,
                        T = T,
                        model_name=f"CH_{model_type}_{obj_type}_{ch_method}_storage_{storage_id}",
                        config={"solve_method": "CH", "model_type": model_type, "ch_method": ch_method, "obj_type": obj_type, "solver": solver,
                                "appointed_pdis": appointed_pdis,
                                "appointed_pch": appointed_pch,
                                "appointed_e": appointed_e}
                    )

        if run_lp:
            schedules["LP"] = build_schedule(
                storage = storage,
                price_ch = price_ch if obj_type == "Profit" else None,
                price_dis = price_dis if obj_type == "Profit" else None,
                signal = selected_signal if obj_type == "SetPointTracking" else None,
                T = T,
                model_name=f"LP_{model_type}_{obj_type}_storage_{storage_id}",
                config={"solve_method": "LP", "model_type": model_type, "obj_type": obj_type, "solver": solver}
            )
    else:
        for schedule in schedules.values():
            schedule.resolve(
                signal=selected_signal if obj_type == "SetPointTracking" else None,
                price_ch=price_ch if obj_type == "Profit" else None,
                price_dis=price_dis if obj_type == "Profit" else None
            )

    results = []

    if run_milp:
        milp_objective = schedules["MILP"].objective_value
    else:
        milp_objective = np.nan

    def add_result(method, schedule):
        simultaneous_product = (
            schedule.es_model.result_pdis * schedule.es_model.result_pch
        )
        violation_mask = np.abs(simultaneous_product) > 1e-4
        simultaneous_count = int(np.sum(violation_mask))
        simultaneous_percentage = simultaneous_count / T * 100
        sci = np.sum(np.sqrt(np.abs(simultaneous_product[violation_mask]))) / (
            T * np.sqrt(storage.pch_max * storage.pdis_max)
        )
        gap_vs_milp = (
            (schedule.objective_value - milp_objective)
            / max(abs(milp_objective), 1e-9) * 100
        )
        results.append([
            storage_id,
            day,
            solver,
            model_type,
            obj_type,
            method,
            schedule.objective_value,
            milp_objective,
            gap_vs_milp,
            schedule.solve_time,
            simultaneous_count,
            simultaneous_percentage,
            sci
        ])

    for method, schedule in schedules.items():
        add_result(method, schedule)

    result_df = pd.DataFrame(
        results,
        columns=["Storage ID", "Day", "Solver", "Model Type", "Objective Type", "Method", "Objective", "MILP Objective", "Gap vs MILP (%)", "Time (s)", "Simultaneous Count", "Simultaneous Percentage (%)", "SCI"]
    )
    print(result_df.to_string(index=False))
    return result_df


def build_dispatch_results(schedules, storage_id, day, signal, T):
    variable_names = [
        "xdis", "xch", "xidle", "pdis", "pch", "e",
        "v_dis2ch", "v_dis2idle", "v_ch2dis",
        "v_ch2idle", "v_idle2dis", "v_idle2ch",
    ]
    dispatch_frames = []

    for method, schedule in schedules.items():
        es_model = schedule.es_model
        dispatch_data = {
            "Storage ID": storage_id,
            "Day": day,
            "Solver": schedule.config["solver"],
            "Model Type": schedule.config["model_type"],
            "Objective Type": schedule.config["obj_type"],
            "Method": method,
            "t": np.arange(T),
            "Signal": signal,
        }
        for name in variable_names:
            values = getattr(es_model, f"result_{name}", None)
            dispatch_data[name] = values if values is not None else np.full(T, np.nan)

        dispatch_frames.append(pd.DataFrame(dispatch_data))

    return pd.concat(dispatch_frames, ignore_index=True)


def main(run_dir):
    data_path = "pozodata"
    data = data_read.DataReader_pozo(data_path)

    summary_path = os.path.join(run_dir, "experiment_summary.csv")
    dispatch_path = os.path.join(run_dir, "dispatch_results.csv")
    manifest_path = os.path.join(run_dir, "sampling_manifest.csv")
    first_result = True

    # All ESS parameter sets
    num_es = len(data.storages)
    num_days = data.signal.shape[0]
    selected_storage_ids = np.arange(
        1, num_es + 1
    )

    # All daily scenarios from 2018 and 2019
    selected_days = np.arange(
        1, num_days + 1
    )

    sampling_manifest = pd.concat([
        pd.DataFrame({"Sample Type": "Storage ID", "Selected Value": selected_storage_ids}),
        pd.DataFrame({"Sample Type": "Day", "Selected Value": selected_days})
    ], ignore_index=True)
    sampling_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"Selected storage IDs: {selected_storage_ids.tolist()}")
    print(f"Selected days: {selected_days.tolist()}")

    for storage_id in selected_storage_ids:
        schedules = {}
        for day in selected_days:
            print("\n" + "=" * 90)
            print(f"Storage ID: {storage_id} | Day: {day}")
            print("=" * 90)
            result_df = run_test(
                data=data,
                storage_id=storage_id,
                day=day,
                schedules=schedules,
                output_dir=run_dir,
            )
            result_df.to_csv(
                summary_path,
                mode="w" if first_result else "a",
                header=first_result,
                index=False,
                encoding="utf-8-sig" if first_result else "utf-8"
            )
            dispatch_df = build_dispatch_results(
                schedules=schedules,
                storage_id=storage_id,
                day=day,
                signal=data.signal[day - 1, :],
                T=data.T,
            )
            dispatch_df.to_csv(
                dispatch_path,
                mode="w" if first_result else "a",
                header=first_result,
                index=False,
                encoding="utf-8-sig" if first_result else "utf-8",
            )
            first_result = False

    print(f"\nAll results saved to: {summary_path}")
    print(f"Dispatch results saved to: {dispatch_path}")
    print(f"Sampling manifest saved to: {manifest_path}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", timestamp)
    os.makedirs(run_dir, exist_ok=False)
    log_path = os.path.join(run_dir, "run.log")

    with open(log_path, "w", encoding="utf-8-sig", buffering=1) as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), \
                contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
            print(f"Run output directory: {run_dir}")
            main(run_dir)
