from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from whalenet_exact_staged import (
    ExactMelComputer,
    ExactWSTComputer,
    ExactResNet,
    ExactMLP,
    class_count_from_manifest,
    count_params,
    evaluate_tensor_model,
    extract_features,
    metrics_from_logits,
    save_json,
    save_model_result,
    train_classifier,
    write_csv,
)


def _row(name: str, data: dict[str, Any], ckpt: Path, best_epoch: str = 'final') -> dict[str, Any]:
    metrics = data['test']['metrics']
    return {
        'model': name,
        'parameters': int(data.get('params', 0)),
        'best_epoch': best_epoch,
        'test_loss': metrics.get('loss', 0.0),
        'test_accuracy': metrics.get('accuracy', 0.0),
        'test_macro_f1': metrics.get('macro_f1', 0.0),
        'test_weighted_f1': metrics.get('weighted_f1', 0.0),
        'test_micro_f1': metrics.get('micro_f1', 0.0),
        'checkpoint': str(ckpt),
    }


def run_whalenet_paper_staged(
    train_manifest: Path,
    test_manifest: Path,
    classes_csv: Path,
    out_dir: Path,
    sample_rate: int = 47600,
    signal_len: int = 8000,
    J: int = 6,
    Q: int = 16,
    hard_lambda: float = 1.0 / 3.0,
    branch_epochs: int = 100,
    fusion_epochs: int = 500,
    batch_size: int = 128,
    feature_batch_size: int = 256,
    num_workers: int = 4,
    device: str = 'cuda',
    seed: int = 0,
    standardize_signal: bool = True,
    include_branch_ablation: bool = False,
) -> list[dict[str, Any]]:
    """Paper-reported WhaleNet baseline, staged and external-audio-backend-free.

    This is the baseline the user asked for after selecting option B:
    paper-reported settings rather than the pasted notebook defaults.

    Main output row: whalenet_paper_hard
      - fixed sample rate: 47600 by default
      - centered crop/pad: 8000 samples
      - signal standardization: on by default, as described in the paper
      - Kymatio Scattering1D with J=6, Q=16 by default
      - ResNet branches: Mel, WST order 1, WST order 2
      - WST1+WST2 MLP fusion
      - final hard convex merge with Mel weight lambda=1/3 by default

    No the original audio backend is imported. Audio loading/resampling/Mel are reimplemented with
    soundfile/wave/scipy/PyTorch utilities from the surrounding environment.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes, class_names = class_count_from_manifest(classes_csv)
    hard_lambda = float(hard_lambda)

    config = {
        'baseline': 'whalenet_paper_staged_the original audio backend_free',
        'main_reported_row': 'whalenet_paper_hard',
        'uses_kymatio_scattering1d': True,
        'uses_the original audio backend': False,
        'fixed_sample_rate': int(sample_rate),
        'signal_len': int(signal_len),
        'J': int(J),
        'Q': int(Q),
        'standardize_signal': bool(standardize_signal),
        'hard_lambda_mel_weight': hard_lambda,
        'branch_epochs': int(branch_epochs),
        'fusion_epochs': int(fusion_epochs),
        'batch_size': int(batch_size),
        'feature_batch_size': int(feature_batch_size),
        'include_branch_ablation_in_main_summary': bool(include_branch_ablation),
        'note': (
            'Paper-reported WhaleNet configuration: fixed-rate preprocessing, 8000-sample centered signals, '
            'standardization, Kymatio WST J=6 Q=16 by default, separate Mel/WST1/WST2 ResNet branches, '
            'WST1+WST2 MLP, and final hard convex merge lambda*Mel + (1-lambda)*WST12. The original audio backend is not used because the user requested no the original audio backend.'
        ),
    }
    save_json(config, out_dir / 'whalenet_paper_config.json')

    print('\n=== WhaleNet paper baseline: feature extraction ===', flush=True)
    mel_front = ExactMelComputer(sample_rate, signal_len, 64, 400, 200, standardize_signal=standardize_signal).to(dev)
    wst1_front = ExactWSTComputer(sample_rate, signal_len, J, Q, 1, standardize_signal=standardize_signal).to(dev)
    wst2_front = ExactWSTComputer(sample_rate, signal_len, J, Q, 2, standardize_signal=standardize_signal).to(dev)

    x_mel_tr, y_tr, sr_tr, _, _ = extract_features(mel_front, train_manifest, feature_batch_size, num_workers, dev, 'extract whalenet mel train')
    x_mel_te, y_te, sr_te, _, _ = extract_features(mel_front, test_manifest, feature_batch_size, num_workers, dev, 'extract whalenet mel test')
    x_w1_tr, _, _, _, _ = extract_features(wst1_front, train_manifest, feature_batch_size, num_workers, dev, 'extract whalenet wst1 train')
    x_w1_te, _, _, _, _ = extract_features(wst1_front, test_manifest, feature_batch_size, num_workers, dev, 'extract whalenet wst1 test')
    x_w2_tr, _, _, _, _ = extract_features(wst2_front, train_manifest, feature_batch_size, num_workers, dev, 'extract whalenet wst2 train')
    x_w2_te, _, _, _, _ = extract_features(wst2_front, test_manifest, feature_batch_size, num_workers, dev, 'extract whalenet wst2 test')

    outputs: dict[str, dict[str, Any]] = {}
    internal_rows: list[dict[str, Any]] = []

    def train_branch(name: str, xtr: torch.Tensor, xte: torch.Tensor) -> torch.nn.Module:
        print(f'\n=== WhaleNet paper baseline: train {name} ResNet branch ===', flush=True)
        model = ExactResNet(num_classes)
        res = train_classifier(model, xtr, y_tr, xte, y_te, branch_epochs, batch_size, dev, lr=0.01, weight_decay=0.001, adamw=True)
        trained = res['model']
        eval_train = evaluate_tensor_model(trained, xtr, y_tr, batch_size, dev)
        outputs[name] = {'test': res, 'train_eval': eval_train, 'params': count_params(trained), 'model': trained}
        save_model_result(name, outputs[name], y_te, sr_te, class_names, out_dir, seed)
        ckpt = out_dir / name / 'best.pt'
        torch.save({'model': trained.state_dict(), 'config': config, 'metrics': res['metrics']}, ckpt)
        internal_rows.append(_row(name, outputs[name], ckpt))
        return trained

    train_branch('whalenet_paper_mel_branch', x_mel_tr, x_mel_te)
    train_branch('whalenet_paper_wst1_branch', x_w1_tr, x_w1_te)
    train_branch('whalenet_paper_wst2_branch', x_w2_tr, x_w2_te)

    print('\n=== WhaleNet paper baseline: train WST1+WST2 MLP ===', flush=True)
    wst_tr = torch.cat([outputs['whalenet_paper_wst1_branch']['train_eval']['logits'], outputs['whalenet_paper_wst2_branch']['train_eval']['logits']], dim=1)
    wst_te = torch.cat([outputs['whalenet_paper_wst1_branch']['test']['logits'], outputs['whalenet_paper_wst2_branch']['test']['logits']], dim=1)
    wst12_mlp = ExactMLP(2 * num_classes, num_classes)
    res_wst12 = train_classifier(wst12_mlp, wst_tr, y_tr, wst_te, y_te, fusion_epochs, batch_size, dev, lr=0.001, weight_decay=0.0, adamw=False)
    eval_wst12_tr = evaluate_tensor_model(res_wst12['model'], wst_tr, y_tr, batch_size, dev)
    outputs['whalenet_paper_wst12_mlp'] = {'test': res_wst12, 'train_eval': eval_wst12_tr, 'params': count_params(res_wst12['model']), 'model': res_wst12['model']}
    save_model_result('whalenet_paper_wst12_mlp', outputs['whalenet_paper_wst12_mlp'], y_te, sr_te, class_names, out_dir, seed)
    ckpt_wst = out_dir / 'whalenet_paper_wst12_mlp' / 'best.pt'
    torch.save({'model': res_wst12['model'].state_dict(), 'config': config, 'metrics': res_wst12['metrics']}, ckpt_wst)
    internal_rows.append(_row('whalenet_paper_wst12_mlp', outputs['whalenet_paper_wst12_mlp'], ckpt_wst))

    print('\n=== WhaleNet paper baseline: final hard Mel+WST merge ===', flush=True)
    star_te = res_wst12['logits']
    star_tr = eval_wst12_tr['logits']
    mel_te = outputs['whalenet_paper_mel_branch']['test']['logits']
    mel_tr = outputs['whalenet_paper_mel_branch']['train_eval']['logits']
    hard_logits_te = hard_lambda * mel_te + (1.0 - hard_lambda) * star_te
    hard_metrics = metrics_from_logits(hard_logits_te, y_te, num_classes)
    hard_emb = torch.cat([star_te, mel_te], dim=1)
    total_params = (
        outputs['whalenet_paper_mel_branch']['params']
        + outputs['whalenet_paper_wst1_branch']['params']
        + outputs['whalenet_paper_wst2_branch']['params']
        + outputs['whalenet_paper_wst12_mlp']['params']
    )
    main = {
        'test': {'logits': hard_logits_te, 'embedding': hard_emb, 'metrics': hard_metrics},
        'train_logits_used_for_lambda': {'wst': star_tr, 'mel': mel_tr},
        'params': int(total_params),
        'lambda_mel_weight': hard_lambda,
    }
    save_model_result('whalenet_paper_hard', main, y_te, sr_te, class_names, out_dir, seed)
    ckpt_main = out_dir / 'whalenet_paper_hard' / 'best.pt'
    torch.save({'config': config, 'metrics': hard_metrics, 'lambda_mel_weight': hard_lambda}, ckpt_main)
    main_row = _row('whalenet_paper_hard', main, ckpt_main)

    fields = ['model','parameters','best_epoch','test_loss','test_accuracy','test_macro_f1','test_weighted_f1','test_micro_f1','checkpoint']
    write_csv(out_dir / 'whalenet_paper_internal_branch_metrics.csv', internal_rows, fields)
    write_csv(out_dir / 'whalenet_paper_main_result.csv', [main_row], fields)
    if include_branch_ablation:
        rows = internal_rows + [main_row]
    else:
        rows = [main_row]
    write_csv(out_dir / 'whalenet_paper_results_for_main_summary.csv', rows, fields)
    return rows
