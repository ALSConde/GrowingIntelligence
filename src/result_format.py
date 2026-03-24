import pathlib
import json


def get_results_json(path: str | pathlib.Path):
    if isinstance(path, str):
        path = pathlib.Path(path)

    r = path.glob("*.json")
    results = sorted(path.glob("*.json"), key=lambda p: int(p.stem.split("_")[-1]))

    return results


def format_results(results):
    formatted = []
    for result in results:
        with open(result, "r") as f:
            data = json.load(f)

        def format_metric_dict(metric_data):
            if not metric_data:
                return {}

            return {k: format(v * 100, ".2f") for k, v in metric_data.items()}

        formatted.append(
            {
                "final_accuracy": format(data.get("final_accuracy", 0) * 100, ".2f"),
                "per_experience_accuracy": format_metric_dict(
                    data.get("per_experience_accuracy")
                ),
                "forgetting": format_metric_dict(data.get("forgetting")),
                "bwt": format_metric_dict(data.get("bwt")),
            }
        )

    return formatted


def main():
    # results_dir = pathlib.Path("./results/CIFAR100_Pretrained/MLP/AGEM")
    # results_dir = pathlib.Path("./results/CIFAR100_Pretrained/WD/AGEM")
    results_dir = pathlib.Path("./results/CIFAR100_Pretrained/MLP_attention/AGEM")
    # results_dir = pathlib.Path("./results/CIFAR100_Pretrained/WD_attention/AGEM")

    results = format_results(get_results_json(results_dir))
    i = 1
    for result in results:
        print(f"Run {i}:")
        print(json.dumps(result, indent=2))
        i += 1


if __name__ == "__main__":
    main()
