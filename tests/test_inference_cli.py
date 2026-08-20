from evaluation.run_inference import build_parser


def test_inference_cli_requires_model_and_input_paths():
    parser = build_parser()
    args = parser.parse_args([
        "--ckpt", "maps.pt",
        "--student", "student.pt",
        "--teacher", "teacher.pt",
        "--camera", "camera.jpg",
        "--manual", "manual.jpg",
        "--out-dir", "out",
    ])
    assert args.ckpt == "maps.pt"
    assert args.student == "student.pt"
    assert args.teacher == "teacher.pt"
    assert args.camera == "camera.jpg"
    assert args.manual == "manual.jpg"
