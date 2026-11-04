"""Etapa 8: a grade pula o que já tem "fim" e roda o resto, na ordem certa."""

import json

from scripts import run_grid


def test_pula_rodadas_completas(tmp_path, monkeypatch):
    monkeypatch.setattr(run_grid, "TREINOS", tmp_path / "train")
    monkeypatch.setattr(run_grid, "RESULTS", tmp_path)
    chamadas = []
    monkeypatch.setattr(run_grid, "treina", lambda seed, mag, div: chamadas.append(("treina", seed, mag, div)))
    monkeypatch.setattr(run_grid, "avalia", lambda seed, mag: chamadas.append(("avalia", seed, mag)))

    def grava(caminho, fim=True):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps({"inicio": "x"} | ({"fim": "y"} if fim else {})))

    # semente 0 a 40×: completa; a 100×: "pac" pronto, "img" interrompido, sem avaliação
    grava(tmp_path / "train" / "40_pac_seed0" / "config.json")
    grava(tmp_path / "train" / "40_img_seed0" / "config.json")
    grava(tmp_path / "mtst" / "40_seed0_config.json")
    grava(tmp_path / "train" / "100_pac_seed0" / "config.json")
    grava(tmp_path / "train" / "100_img_seed0" / "config.json", fim=False)

    run_grid.roda_grade(seeds=[0, 10], mags=[40, 100])
    assert chamadas == [
        ("treina", 0, 100, "img"), ("avalia", 0, 100),
        ("treina", 10, 40, "pac"), ("treina", 10, 40, "img"), ("avalia", 10, 40),
        ("treina", 10, 100, "pac"), ("treina", 10, 100, "img"), ("avalia", 10, 100),
    ]
