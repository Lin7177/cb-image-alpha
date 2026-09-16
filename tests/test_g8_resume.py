"""Synthetic recovery-state equivalence, no real samples."""

import copy

import torch


def test_adam_dropout_shuffle_resume_equivalence():
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        torch.manual_seed(12)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 5), torch.nn.Dropout(0.5), torch.nn.Linear(5, 1)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        generator = torch.Generator().manual_seed(42)
        x = torch.arange(24, dtype=torch.float32).reshape(8, 3).to(device) / 24

        def step(m, o, g, x=x):
            ids = torch.randperm(8, generator=g)
            o.zero_grad()
            m(x[ids]).square().mean().backward()
            o.step()

        step(model, optimizer, generator)
        saved = copy.deepcopy(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "shuffle": generator.get_state(),
            }
        )
        step(model, optimizer, generator)
        expected = copy.deepcopy(model.state_dict())
        restored = torch.nn.Sequential(
            torch.nn.Linear(3, 5), torch.nn.Dropout(0.5), torch.nn.Linear(5, 1)
        ).to(device)
        other = torch.optim.Adam(restored.parameters(), lr=1e-5)
        restored.load_state_dict(saved["model"])
        other.load_state_dict(saved["optimizer"])
        g = torch.Generator()
        torch.set_rng_state(saved["cpu"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda"])
        g.set_state(saved["shuffle"])
        step(restored, other, g)
        for name, value in restored.state_dict().items():
            assert torch.equal(value, expected[name])
