"""驗證 Stage 2 辨識含 gender 的 Stage 1 checkpoint 維度。"""

import unittest

import torch

from model_stage2realfake import infer_checkpoint_dims, prepare_stage1_state
from prosody_utils import infer_checkpoint_prosody_dim


class Stage2CheckpointDimensionTests(unittest.TestCase):
    def test_infers_gender_separately_from_prosody(self):
        for out_dim, expected in ((320, (128, 0)), (322, (128, 2)), (448, (256, 0)), (450, (256, 2))):
            with self.subTest(out_dim=out_dim):
                self.assertEqual(infer_checkpoint_dims({"final_proj.weight": torch.zeros(out_dim, 8)}), expected)

    def test_enabled_gender_rejects_a_stage1_checkpoint_without_gender(self):
        with self.assertRaisesRegex(ValueError, "requires a Stage-1 checkpoint with gender"):
            prepare_stage1_state({"final_proj.weight": torch.zeros(448, 8)}, 256, use_gender=True)

    def test_prepares_prosody_dimension_with_or_without_appended_gender(self):
        for out_dim, prosody_dim in ((320, 128), (322, 128), (448, 256), (450, 256)):
            for with_layernorm in (False, True):
                with self.subTest(out_dim=out_dim, with_layernorm=with_layernorm):
                    state = {"final_proj.weight": torch.zeros(out_dim, 8)}
                    if with_layernorm:
                        state["pros_ln.weight"] = torch.ones(prosody_dim)

                    prepared = prepare_stage1_state(state, prosody_dim)

                    self.assertEqual(infer_checkpoint_prosody_dim(prepared), prosody_dim)

    def test_infers_prosody_dimension_from_layernorm_only(self):
        self.assertEqual(infer_checkpoint_prosody_dim({"pros_ln.weight": torch.ones(256)}), 256)

    def test_rejects_unsupported_projection_dimensions(self):
        for out_dim in (319, 321, 323, 449, 451):
            with self.subTest(out_dim=out_dim):
                with self.assertRaisesRegex(ValueError, "Unsupported checkpoint prosody dim"):
                    prepare_stage1_state({"final_proj.weight": torch.zeros(out_dim, 8)}, 128)

    def test_rejects_disagreement_between_projection_and_layernorm(self):
        with self.assertRaisesRegex(ValueError, "mismatch between final_proj and pros_ln"):
            prepare_stage1_state({
                "final_proj.weight": torch.zeros(322, 8), "pros_ln.weight": torch.ones(256),
            }, 128)

    def test_rejects_checkpoint_without_dimension_information(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer prosody dim"):
            infer_checkpoint_prosody_dim({})

    def test_prepares_projection_without_gender_rows(self):
        for prosody_dim, out_dim in ((128, 320), (256, 448)):
            for use_gender in (False, True):
                with self.subTest(prosody_dim=prosody_dim, use_gender=use_gender):
                    state = {
                        "final_proj.weight": torch.full((out_dim, 8), 3.0),
                        "final_proj.bias": torch.full((out_dim,), 4.0),
                        "pros_ln.weight": torch.ones(prosody_dim),
                    }
                    if use_gender:
                        state["final_proj.weight"] = torch.cat([
                            state["final_proj.weight"], torch.full((2, 8), 9.0),
                        ])
                        state["final_proj.bias"] = torch.cat([
                            state["final_proj.bias"], torch.full((2,), 10.0),
                        ])

                    actual = prepare_stage1_state(state, prosody_dim)

                    torch.testing.assert_close(actual, {
                        "final_proj.weight": torch.full((out_dim, 8), 3.0),
                        "final_proj.bias": torch.full((out_dim,), 4.0),
                        "pros_ln.weight": torch.ones(prosody_dim),
                    })

    def test_preparing_checkpoint_does_not_change_original_state(self):
        state = {"final_proj.weight": torch.ones(322, 8), "final_proj.bias": torch.ones(322)}
        original = {key: value.clone() for key, value in state.items()}

        prepare_stage1_state(state, 128)

        torch.testing.assert_close(state, original, rtol=0, atol=0)

    def test_prepares_wrapped_checkpoint_with_distributed_prefix(self):
        checkpoint = {"state_dict": {
            "module.final_proj.weight": torch.ones(322, 8),
            "module.final_proj.bias": torch.zeros(322),
            "module.ssl.weight": torch.full((8, 8), 7.0),
        }}

        actual = prepare_stage1_state(checkpoint, 128)

        torch.testing.assert_close(actual, {
            "final_proj.weight": torch.ones(320, 8),
            "final_proj.bias": torch.zeros(320),
            "ssl.weight": torch.full((8, 8), 7.0),
        })

    def test_rejects_prosody_mismatch_instead_of_truncating_other_features(self):
        with self.assertRaisesRegex(ValueError, "Stage-1 checkpoint has 256, Stage-2 expects 128"):
            prepare_stage1_state({"final_proj.weight": torch.zeros(450, 8)}, 128)


if __name__ == "__main__":
    unittest.main()
