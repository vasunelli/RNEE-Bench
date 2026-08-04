import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "rnee_build" / "00_preflight.py"
SPEC = importlib.util.spec_from_file_location("rnee_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_field_contract_separates_road_class_and_speed_source():
    contract = load_yaml(ROOT / "configs" / "rnee_build" / "field_contract.yaml")
    result = MODULE.validate_field_contract(contract)
    assert result["passed"]
    assert result["checks"]["class_of_speed_limit_mapping_forbidden"]
    assert result["checks"]["road_and_speed_namespaces_disjoint"]


def test_legacy_eved_fields_are_reference_only_and_not_l0_inputs():
    base = load_yaml(ROOT / "configs" / "rnee_build" / "base.yaml")
    contract = load_yaml(ROOT / "configs" / "rnee_build" / "field_contract.yaml")
    assert base["source_policy"]["l0_inputs"] == [
        "official_VED_dynamic_data",
        "official_VED_static_data",
    ]
    assert base["source_policy"]["eved_role"] == "differential_audit_only"
    assert "Energy_Consumption" in contract["legacy_reference_only_fields"]
    assert "Energy_Consumption" in contract["prohibited_model_features"]


def test_osm_contract_is_historical_only():
    base = load_yaml(ROOT / "configs" / "rnee_build" / "base.yaml")
    assert base["osm"]["temporal_policy"] == "historical_snapshots_only"
    assert {item["snapshot_date"] for item in base["osm"]["snapshots"]} == {
        "2017-01-01",
        "2018-01-01",
    }
    assert all("latest" not in item["url"] for item in base["osm"]["snapshots"])


def test_runtime_versions_and_commits_are_exact():
    base = load_yaml(ROOT / "configs" / "rnee_build" / "base.yaml")
    assert base["runtime"]["required_python_major_minor"] == "3.11"
    assert base["ved"]["commit"] == "6baa4963782d515a67d32a5490bd5d11f5d9bf0d"
    assert base["valhalla"]["version"] == "3.7.0"
    assert (
        base["valhalla"]["commit"]
        == "72f459fc5661fb906ad424be5378c4e32d9a5b3b"
    )
    assert len(base["valhalla"]["wheel_sha256"]) == 64
    assert base["valhalla"]["wheel_url"].startswith(
        "https://files.pythonhosted.org/"
    )
    assert base["valhalla"]["source_tag_ref"] == "refs/tags/3.7.0"


def test_qa_contract_includes_hard_conservation_and_leakage_gates():
    qa = load_yaml(ROOT / "configs" / "rnee_build" / "qa_thresholds.yaml")
    assert qa["row_assembly"]["input_output_row_ratio"] == 1.0
    assert qa["row_assembly"]["duplicate_row_id_count"] == 0
    assert qa["map_matching"]["matched_point_cardinality"][
        "pass_equal_input"
    ]
    assert qa["segments_and_splits"]["split_entity_overlap_max"] == 0
    assert qa["segments_and_splits"]["test_evaluation_count"] == 1


def test_direct_requirements_are_exactly_pinned():
    requirements = MODULE.parse_pinned_requirements(
        ROOT / "configs" / "rnee_build" / "requirements-rnee.in"
    )
    assert requirements
    assert requirements["numpy"] == "2.3.4"
    assert requirements["osmium"] == "4.2.0"


def test_missing_lock_package_returns_false_without_crashing():
    results, passed = MODULE.resolved_lock_status(
        {"definitely-not-an-installed-rnee-package": "0.0.0"}
    )
    assert not passed
    assert results["definitely-not-an-installed-rnee-package"]["actual"] is None


def test_missing_target_files_return_failed_status_without_crashing(tmp_path):
    result = MODULE.target_gate_status(
        tmp_path / "missing-summary.json",
        tmp_path / "missing-contract.json",
        {"PASS_OPERATIONAL_TARGET_ONLY"},
    )
    assert not result["accepted"]
    assert not result["summary_exists"]
    assert not result["contract_exists"]
    assert result["contract_sha256"] is None


def test_fingerprint_changes_when_transitive_lock_changes():
    base = load_yaml(ROOT / "configs" / "rnee_build" / "base.yaml")
    direct = {"numpy": {"actual": "2.3.4"}}
    lock_a = {"urllib3": {"actual": "2.7.0"}}
    lock_b = {"urllib3": {"actual": "2.7.1"}}
    valhalla = {"actual": "3.7.0"}
    hashes = {"dependency_lock_sha256": "a" * 64}
    first = MODULE.build_fingerprint(base, direct, lock_a, valhalla, hashes)
    second = MODULE.build_fingerprint(base, direct, lock_b, valhalla, hashes)
    assert first["stable_sha256"] != second["stable_sha256"]


def test_license_manifest_inventories_all_locked_dependencies():
    base = load_yaml(ROOT / "configs" / "rnee_build" / "base.yaml")
    lock = MODULE.parse_pinned_requirements(
        ROOT / "configs" / "rnee_build" / "requirements-rnee.lock.txt"
    )
    manifest = MODULE.license_manifest(base, lock)
    assert manifest["complete"]
    assert manifest["python_dependency_inventory_complete"]
    assert len(manifest["python_dependency_records"]) == len(lock)
