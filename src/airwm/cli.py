import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="airwm", description="Aerial RGB-D world-model experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="Check installed versions and CUDA without launching Isaac Sim"
    )
    doctor.add_argument("--output")
    profile = commands.add_parser(
        "profile", help="Measure RSSM GPU training memory with synthetic tensors"
    )
    profile.add_argument("--output", default="runs/profile.json")
    profile.add_argument("--batch-size", type=int, default=16)
    profile.add_argument("--length", type=int, default=64)
    profile.add_argument("--steps", type=int, default=3)
    profile.add_argument("--device", default="cuda:0")
    collect = commands.add_parser(
        "collect", help="Fly a drone in the local room and record RGB-D episodes"
    )
    collect.add_argument("--output", default="data/pilot")
    collect.add_argument("--episodes", type=int, default=3)
    collect.add_argument("--seconds", type=float, default=30.0)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--trajectory", choices=["hover", "sweep", "maneuvers"], default="sweep")
    collect.add_argument("--device", default="cuda:0")
    collect.add_argument("--gui", action="store_true")
    branch = commands.add_parser(
        "branch-collect", help="Replay matched motor-command interventions"
    )
    branch.add_argument("--source", required=True, help="Folder of complete reference flights")
    branch.add_argument("--output", required=True, help="Fresh folder for evaluation-only branches")
    branch.add_argument("--sources", type=int, default=8)
    branch.add_argument("--start-frame", type=int, default=160)
    branch.add_argument("--horizon", type=int, default=40)
    branch.add_argument("--seed", type=int, default=101)
    branch.add_argument("--device", default="cuda:0")
    branch.add_argument("--gui", action="store_true")
    from airwm.branching import INTERVENTIONS

    branch.add_argument(
        "--branches", nargs="+", choices=list(INTERVENTIONS), default=list(INTERVENTIONS)
    )
    branch.set_defaults(trajectory="replay")
    branch_eval = commands.add_parser(
        "branch-evaluate", help="Evaluate paired simulated action responses"
    )
    branch_eval.add_argument("--data", required=True)
    branch_eval.add_argument("--checkpoint", required=True)
    branch_eval.add_argument("--output", required=True)
    branch_eval.add_argument("--context", type=int, default=20)
    branch_eval.add_argument("--device", default="cuda:0")
    train = commands.add_parser("train", help="Train the RSSM from complete flight files")
    train.add_argument("--data", default="data/pilot")
    train.add_argument("--run", default="runs/baseline")
    train.add_argument("--steps", type=int, default=10000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--workers", type=int, default=0, help="CPU data-loading workers")
    train.add_argument("--length", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--tiny", action="store_true", help="Small model for pipeline tests only")
    evaluate = commands.add_parser(
        "evaluate", help="Predict held-out flights without future observations"
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", default="runs/evaluation")
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument(
        "--split-file",
        help="Evaluate a shared split.json; rejects this checkpoint's training flights",
    )
    evaluate.add_argument("--context", type=int, default=20)
    evaluate.add_argument("--horizon", type=int, default=40)
    evaluate.add_argument("--windows", type=int, default=10)
    evaluate.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from airwm.diagnostics import doctor

        return doctor(args)
    if args.command == "profile":
        from airwm.profile import profile

        profile(args)
        return 0
    if args.command in ("collect", "branch-collect"):
        if args.command == "branch-collect" and (
            min(args.sources, args.horizon) < 1
            or args.start_frame < 19
            or len(set(args.branches)) != len(args.branches)
        ):
            parser.error("Use positive sources/horizon, start-frame >= 19, and unique branches")
        from importlib.metadata import version

        if version("isaaclab").split("+")[0] != "3.0.0":
            parser.error("Collection requires the pinned Isaac Lab 3.0.0 airwm environment")
        from airwm.simulation import collect

        collect(args)
    elif args.command == "branch-evaluate":
        if args.context < 1:
            parser.error("Context must be positive")
        from airwm.branch_evaluate import evaluate_branches

        evaluate_branches(args)
    elif args.command == "train":
        if args.eval_every < 1:
            parser.error("--eval-every must be positive")
        if args.workers < 0:
            parser.error("--workers must be nonnegative")
        from airwm.train import train

        train(args)
    else:
        if min(args.context, args.horizon, args.windows) < 1:
            parser.error("Context, horizon and windows must be positive")
        from airwm.evaluate import evaluate

        evaluate(args)
    return 0
