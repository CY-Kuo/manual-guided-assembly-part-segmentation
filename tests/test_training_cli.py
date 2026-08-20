from training.train import ABLATION_CHOICES, build_parser


def test_training_cli_parses_final_ablation_choices():
    assert ABLATION_CHOICES == (
        "base",
        "fuse_10_aux",
        "fuse_6_10_aux",
        "fuse_6_aux",
        "fuse_aux_only",
    )

    parser = build_parser()
    args = parser.parse_args([
        "--config", "training/config.yaml",
        "--ablation", "fuse_aux_only",
        "--fold-root", "/data/fold3",
        "--out-dir", "runs/fold3/fuse_aux_only",
    ])
    assert args.ablation == "fuse_aux_only"
    assert args.fold_root == "/data/fold3"
