import copy
import unittest
from unittest.mock import patch

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from model_stage1real import ProSDDStage1
from model_stage2realfake import ProSDDStage2
from main_eval import inference_forward
from multi_gpu import place_model, resolve_devices


def tiny_backbone(*args, **kwargs):
    return Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=16, conv_dim=(8,), conv_stride=(2,),
        conv_kernel=(3,), num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True,
        hidden_dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
    ))


def build(stage, pool="mean"):
    with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
        cls = ProSDDStage1 if stage == 1 else ProSDDStage2
        kwargs = {} if stage == 1 else {"classifier_pool": pool}
        return cls(mask_prob=0.01, mask_span_len=1,
                   num_time_neg=2, num_spk_neg=2, **kwargs)


def objective(out, labels):
    if isinstance(out, torch.Tensor):
        return out
    return out["ssl_loss"] + torch.nn.functional.cross_entropy(
        out["logits"], labels, weight=out["logits"].new_tensor([0.1, 0.9]))


class PlacementTests(unittest.TestCase):
    def compare(self, stage, pool, devices, training=True, layerdrop=0.0, freeze=False):
        torch.manual_seed(31)
        baseline = build(stage, pool).to(devices[0])
        baseline.ssl.encoder.config.layerdrop = layerdrop
        if freeze:
            for parameter in baseline.cls_head.parameters():
                parameter.requires_grad = False
            if pool == "attn":
                baseline.attn_q.requires_grad = False
                for parameter in baseline.attn.parameters():
                    parameter.requires_grad = False
        candidate = copy.deepcopy(baseline).cpu()
        place_model(candidate, devices)
        if devices[0].type == "cuda":
            for index, layer in enumerate(candidate.ssl.encoder.layers):
                expected = devices[index * len(devices) // len(candidate.ssl.encoder.layers)]
                self.assertTrue(all(p.device == expected for p in layer.parameters()))
        baseline.train(training)
        candidate.train(training)
        self.assertEqual(list(baseline.state_dict()), list(candidate.state_dict()))
        self.assertEqual([n for n, _ in baseline.named_parameters()],
                         [n for n, _ in candidate.named_parameters()])
        inputs = (torch.randn(3, 31, device=devices[0]),
                  torch.randn(3, 192, device=devices[0]),
                  torch.randn(3, 12, 256, device=devices[0]),
                  torch.tensor([0, 0, 1], device=devices[0]))
        labels = torch.tensor([0, 1, 1], device=devices[0])
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4)
                      for m in (baseline, candidate)]
        outputs = []
        for model, optimizer in zip((baseline, candidate), optimizers):
            torch.manual_seed(47)
            out = model(*inputs)
            outputs.append(out)
            objective(out, labels).backward()
            optimizer.step()
        left = outputs[0] if isinstance(outputs[0], dict) else {"loss": outputs[0]}
        right = outputs[1] if isinstance(outputs[1], dict) else {"loss": outputs[1]}
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=2e-4, atol=2e-5)
        for (name, a), (_, b) in zip(baseline.named_parameters(), candidate.named_parameters()):
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad.cpu(), b.grad.cpu(), rtol=3e-3, atol=2e-5)
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=3e-3, atol=2e-5)
        restored = build(stage, pool)
        restored.load_state_dict(candidate.state_dict(), strict=True)

    def test_layerdrop_and_frozen_classifier(self):
        self.compare(2, "attn", [torch.device("cpu"), torch.device("cpu:0")],
                     layerdrop=1.0, freeze=True)

    def test_evaluation_entrypoint(self):
        for pool in ("mean", "attn"):
            with self.subTest(pool=pool):
                baseline = build(2, pool).eval()
                candidate = copy.deepcopy(baseline)
                place_model(candidate, [torch.device("cpu"), torch.device("cpu:0")])
                wav = torch.randn(3, 31)
                torch.testing.assert_close(inference_forward(baseline, wav),
                                           inference_forward(candidate, wav), rtol=0, atol=0)

    def test_cpu_forward_backward_update_and_checkpoint(self):
        # cpu:0 deliberately exercises remote-layer hooks without CUDA.
        for stage, pool in ((1, "mean"), (2, "mean"), (2, "attn")):
            with self.subTest(stage=stage, pool=pool):
                self.compare(stage, pool, [torch.device("cpu"), torch.device("cpu:0")])

    def test_default_placement(self):
        model = build(1)
        self.assertIs(place_model(model, [torch.device("cpu")]), model)
        self.assertFalse(any(layer._forward_hooks for layer in model.ssl.encoder.layers))

    def test_device_validation(self):
        with patch("torch.cuda.device_count", return_value=2):
            for ids in ([], [0, 0], [-1], [2]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    resolve_devices(ids)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_devices(), [torch.device("cpu")])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_two_gpu_forward_backward_update_and_checkpoint(self):
        # Eval mode disables classifier dropout; autograd remains enabled.
        for stage, pool in ((1, "mean"), (2, "mean"), (2, "attn")):
            with self.subTest(stage=stage, pool=pool):
                self.compare(stage, pool, [torch.device("cuda:0"), torch.device("cuda:1")], False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
