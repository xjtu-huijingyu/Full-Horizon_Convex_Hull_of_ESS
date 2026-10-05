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


def run_test(storage, storage_id, scenario, price_ch=None, price_dis=None, schedules=None, output_dir="output", appointed_pdis=None, appointed_pch=None, appointed_e=None):
    T = len(price_ch)
    solver = "copt"
    model_type = "Basic" 
    obj_type = "Profit"  # Run the OOS test
    run_milp, run_ch, run_lp = 1, 1, 1  # Select which formulations to test

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
                signal = None, 
                T = T, 
                model_name=f"MILP_{model_type}_{obj_type}_storage_{storage_id}", 
                config={"solve_method": "MILP", "model_type": model_type, "obj_type": obj_type, "solver": solver}
            )

        if run_ch:
            schedules["Network CH"] = build_schedule(
                storage = storage,
                price_ch = price_ch if obj_type == "Profit" else None,
                price_dis = price_dis if obj_type == "Profit" else None,
                signal = None,
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
                        signal = None,
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
                signal = None,
                T = T,
                model_name=f"LP_{model_type}_{obj_type}_storage_{storage_id}",
                config={"solve_method": "LP", "model_type": model_type, "obj_type": obj_type, "solver": solver}
            )
    else:
        for schedule in schedules.values():
            schedule.resolve(
                signal=None,
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
            scenario,
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
        columns=["Storage ID", "Scenario", "Solver", "Model Type", "Objective Type", "Method", "Objective", "MILP Objective", "Gap vs MILP (%)", "Time (s)", "Simultaneous Count", "Simultaneous Percentage (%)", "SCI"]
    )
    print(result_df.to_string(index=False))
    return result_df


def build_dispatch_results(schedules, storage_id, scenario, price_ch, price_dis, T):
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
            "Scenario": scenario,
            "Solver": schedule.config["solver"],
            "Model Type": schedule.config["model_type"],
            "Objective Type": schedule.config["obj_type"],
            "Method": method,
            "Scenario": scenario,
            "t": np.arange(T),
            "Price_ch": price_ch,
            "Price_dis": price_dis,
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

    # Use all available cases
    random_seed = 42
    # All ESS parameter sets
    num_es = len(data.storages)
    num_scenarios = data.price_ch.shape[0]
    selected_storage_ids = np.arange(
        1, num_es + 1
    )

    # All daily scenarios from 2018 and 2019
    selected_scenarios = np.arange(
        1, num_scenarios + 1
    )

    sampling_manifest = pd.concat([
        pd.DataFrame({"Sample Type": "Storage ID", "Selected Value": selected_storage_ids}),
        pd.DataFrame({"Sample Type": "Scenario", "Selected Value": selected_scenarios})
    ], ignore_index=True)
    sampling_manifest.insert(0, "Random Seed", random_seed)
    sampling_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"Selected storage IDs: {selected_storage_ids.tolist()}")
    print(f"Selected scenarios: {selected_scenarios.tolist()}")


    for storage_id in selected_storage_ids:
        schedules = {}
        for scenario in selected_scenarios:
            print("\n" + "=" * 90)
            print(f"Storage ID: {storage_id} | Scenario: {scenario}")
            print("=" * 90)
            result_df = run_test(
                storage=data.storages[storage_id - 1],
                storage_id=storage_id,
                scenario=scenario,
                price_dis=data.price_dis[scenario - 1, :],
                price_ch=data.price_ch[scenario - 1, :],
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
                scenario=scenario,
                price_ch=data.price_ch[scenario - 1, :],
                price_dis=data.price_dis[scenario - 1, :],
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
