"""Model-level regression tests for deterministic GeoVAE inference and TTLO."""
import unittest

import torch
import torch.nn as nn

from model import (
    ConvVAE3D,
    GeoTransformerBlock,
    LoRAConv3d,
    adapt_to_observation,
    apply_lora_to_model,
)


torch.set_num_threads(1)


def sparse_observation():
    observation = torch.zeros(1, 2, 16, 16, 16)
    observation[0, 1, 2, 2, 2] = 1.0
    observation[0, 0, 2, 2, 2] = 1.0
    observation[0, 1, 12, 12, 12] = 1.0
    return observation


class ModelAdaptationTests(unittest.TestCase):
    def test_relative_3d_bias_changes_attention_and_receives_gradient(self):
        torch.manual_seed(7)
        block = GeoTransformerBlock(
            channels=8, num_heads=2, dropout=0.0,
            use_relative_position=True, max_distance=2,
        ).eval()
        sample = torch.randn(1, 8, 2, 2, 2)
        before = block(sample).detach()
        with torch.no_grad():
            block.relative_bias[0, 0, 3] = 4.0
        after = block(sample)
        self.assertGreater((before - after).abs().max().item(), 1e-5)
        after.square().mean().backward()
        self.assertIsNotNone(block.relative_bias.grad)
        self.assertGreater(block.relative_bias.grad.abs().sum().item(), 0.0)

    def test_evaluation_uses_posterior_mean(self):
        torch.manual_seed(2)
        model = ConvVAE3D(
            grid_size=16, latent_dim=8, base_channels=4,
            use_transformer=True, use_relative_position=True,
        ).eval()
        observation = sparse_observation()
        first, _, _ = model(observation)
        second, _, _ = model(observation)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_convolution_adapters_are_optimized_and_frozen_base(self):
        conv = nn.Conv3d(2, 3, kernel_size=3, padding=1)
        network = nn.Sequential(conv)
        network, parameters = apply_lora_to_model(
            network, rank=2, target_modules=["0"], dropout=0.0,
        )
        self.assertIsInstance(network[0], LoRAConv3d)
        self.assertEqual(len(parameters), 2)
        self.assertFalse(network[0].original_layer.weight.requires_grad)
        input_volume = torch.randn(1, 2, 4, 4, 4)
        network(input_volume).square().mean().backward()
        self.assertGreater(network[0].lora_B.grad.abs().sum().item(), 0.0)

    def test_latent_and_joint_adaptation_are_isolated_and_repeatable(self):
        torch.manual_seed(3)
        model = ConvVAE3D(grid_size=16, latent_dim=8, base_channels=4)
        source = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        observation = sparse_observation()
        probability, latent = adapt_to_observation(
            model, observation, mode="latent", steps=2, patience=None,
        )
        self.assertEqual(tuple(probability.shape), (1, 16, 16, 16))
        self.assertGreater(latent["z_grad_norm_first_step"], 0.0)
        self.assertEqual(latent["lora_trainable_parameters"], 0)
        projected, joint = adapt_to_observation(
            model, observation, mode="joint", steps=2,
            lora_rank=2, patience=None, project_observed=True, seed=11,
        )
        again, again_metadata = adapt_to_observation(
            model, observation, mode="joint", steps=2,
            lora_rank=2, patience=None, project_observed=True, seed=11,
        )
        torch.testing.assert_close(projected, again, rtol=0, atol=0)
        self.assertEqual(joint["best_step"], again_metadata["best_step"])
        self.assertGreater(joint["z_grad_norm_first_step"], 0.0)
        self.assertGreater(joint["lora_grad_norm_first_step"], 0.0)
        self.assertGreater(joint["lora_trainable_parameters"], 0)
        self.assertEqual(joint["obs_mse_post_projection"], 0.0)
        self.assertEqual(projected[0, 2, 2, 2].item(), 1.0)
        self.assertEqual(projected[0, 12, 12, 12].item(), 0.0)
        self.assertTrue(model.training)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, source[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
