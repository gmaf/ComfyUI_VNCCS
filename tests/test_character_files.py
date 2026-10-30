"""Character file listing for clients that mirror VNCCS results (VNCCS Studio)."""

import hashlib

import utils


def test_lists_images_with_view_paths_sizes_hashes_and_archives(monkeypatch, tmp_path):
    root = tmp_path / "output" / "VNCCS" / "Characters"
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(root))
    sprites = root / "Mira" / "Sprites" / "Naked" / "Neutral"
    (sprites / "V1").mkdir(parents=True)
    (sprites / "sprite_pose_0001.png").write_bytes(b"new")
    (sprites / "V1" / "sprite_pose_0001.png").write_bytes(b"old")
    (root / "Mira" / "cache").mkdir()
    (root / "Mira" / "cache" / "preview.png").write_bytes(b"preview")
    (root / "Mira" / "notes.txt").write_text("ignored")

    files = {f["path"]: f for f in utils.list_character_files("Mira")}

    assert sorted(files) == [
        "Sprites/Naked/Neutral/V1/sprite_pose_0001.png",
        "Sprites/Naked/Neutral/sprite_pose_0001.png",
        "cache/preview.png",
    ]
    current = files["Sprites/Naked/Neutral/sprite_pose_0001.png"]
    assert current["subfolder"] == "VNCCS/Characters/Mira/Sprites/Naked/Neutral"
    assert current["filename"] == "sprite_pose_0001.png"
    assert current["type"] == "output"
    assert current["size"] == 3
    assert current["sha256"] == hashlib.sha256(b"new").hexdigest()
    assert current["archived"] is False
    assert files["Sprites/Naked/Neutral/V1/sprite_pose_0001.png"]["archived"] is True


def test_hash_is_recomputed_when_the_file_changes(monkeypatch, tmp_path):
    root = tmp_path / "Characters"
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(root))
    image = root / "Mira" / "cache" / "preview.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"one")
    assert utils.list_character_files("Mira")[0]["sha256"] == hashlib.sha256(b"one").hexdigest()
    image.write_bytes(b"three")
    assert utils.list_character_files("Mira")[0]["sha256"] == hashlib.sha256(b"three").hexdigest()


def test_missing_character_lists_nothing_and_unsafe_names_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    assert utils.list_character_files("Nobody") == []
    try:
        utils.list_character_files("../etc")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe name accepted")
