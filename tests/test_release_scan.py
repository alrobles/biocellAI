"""Secret-scan tests for the HF release packager.

The scanner must catch real credential shapes while staying silent on
ordinary public identifiers (git SHAs, HF revision IDs, sha256 digests)
that fill the release manifests.
"""
import json
import subprocess

from scripts.hf_release import ROOT, scan


def _write(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content)
    return f


def test_git_sha_is_not_flagged(tmp_path):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True).stdout.strip()
    _write(tmp_path, "m.json", json.dumps({"commit": head}))
    assert scan(tmp_path) == []


def test_sha256_digest_is_not_flagged(tmp_path):
    _write(tmp_path, "m.json", json.dumps({"sha256": "a" * 64}))
    _write(tmp_path, "note.txt", "digest " + "0123456789abcdef" * 4)
    assert scan(tmp_path) == []


def test_hf_snapshot_ids_are_not_flagged(tmp_path):
    rev = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
    _write(tmp_path, "a.json", json.dumps({
        "dir": f"hf_cache/hub/models--x--y/snapshots/{rev}/w",
        "snapshot_id": rev, "refs_main": rev}))
    _write(tmp_path, "b.json", json.dumps({
        "path": f"hf_cache/hub/.locks/models--x--y/{rev}.lock"}))
    _write(tmp_path, "c.json", json.dumps({
        "path": f"repos/x/.git/objects/pack/pack-{rev}.idx"}))
    assert scan(tmp_path) == []


def test_bare_40hex_token_is_flagged(tmp_path):
    _write(tmp_path, "notes.txt",
           "release id " + "deadbeef" * 5 + " not a sha context")
    hits = scan(tmp_path)
    assert any("notes.txt" in h for h in hits)


def test_known_token_prefixes_flagged(tmp_path):
    _write(tmp_path, "a.txt", "hf_" + "Ab3" * 10)
    _write(tmp_path, "b.txt", "key: sk-" + "xY9_" * 8)
    _write(tmp_path, "c.txt", "aws AKIA" + "Z7QF" * 4)
    _write(tmp_path, "d.txt", "ghp_" + "z9" * 15)
    _write(tmp_path, "e.txt", "github_pat_" + "a1_" * 10)
    hits = scan(tmp_path)
    assert len(hits) == 5


def test_explicit_credential_assignment_flagged(tmp_path):
    _write(tmp_path, "a.txt", 'api_key = "' + "Kx9" * 10 + '"')
    _write(tmp_path, "b.yaml", "token: " + "m2Vw" * 8)
    hits = scan(tmp_path)
    assert len(hits) == 2
