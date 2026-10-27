"""Delivery integrity and CLI safety tests; no model fitting or frozen output writes."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import project, replay_frozen_business as replay, verify_project_delivery as delivery


class ProjectDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = json.loads((project.ROOT/"docs/project_facts.json").read_text(encoding="utf-8"))

    def test_delivery_files_links_claims_and_available_sources_agree(self):
        result = delivery.verify()
        self.assertEqual(result["status"], "passed", result["errors"])
        self.assertFalse(result["model_training_performed"])
        self.assertFalse(result["historical_audit_files_written"])

    def test_broken_source_link_fails_even_in_source_only_mode(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/"review.md").write_text('[code](scripts/missing.py) [private](artifacts/private.parquet)',encoding="utf-8")
            result = delivery.check_links(root,["review.md"],source_only=True)
            self.assertEqual(len(result["errors"]),1)
            self.assertIn("scripts/missing.py",result["errors"][0])
            self.assertEqual(len(result["unavailable_evidence_links"]),1)

    def test_source_only_copy_passes_docs_review_and_discloses_unavailable_private_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            documents = ["README.md","README.en.md",*[p.relative_to(project.ROOT).as_posix()
                         for p in (project.ROOT/"docs").glob("*.md")],"dashboard/powerbi/README.md"]
            files = {*project.REQUIRED_DOCS,*documents,"requirements.txt","requirements-lock.txt",
                     "docs/figures/architecture.png","tests/test_project_delivery.py",
                     "scripts/verify_project_delivery.py","scripts/render_project_architecture.py"}
            files.update(args[0] for steps in [*project.PIPELINES.values(),*project.VERIFY.values()] for args in steps)
            for document in documents:
                for destination in delivery.markdown_links(project.ROOT/document):
                    target = ((project.ROOT/document).parent/destination).resolve()
                    relative = target.relative_to(project.ROOT).as_posix()
                    if not relative.startswith(("artifacts/","data/","reports/","submissions/")):
                        files.add(relative)
            for relative in files:
                source,target = project.ROOT/relative,root/relative
                if source.is_dir():
                    target.mkdir(parents=True,exist_ok=True)
                elif source.is_file():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,target)
            result = delivery.verify(root)
            self.assertEqual(result["status"],"passed",result["errors"])
            self.assertTrue(result["source_only_copy"])
            self.assertTrue(result["source_provenance"]["unavailable"])
            self.assertFalse(result["source_provenance"]["fact_recollection_performed"])

    def test_markdown_parser_preserves_images_and_ignores_code_and_external_urls(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/"review.md").write_text('![image](fig.png) [guide](<my guide.md>) [web](https://example.com)\n'
                '`[code](ignored.py)`\n```text\n[code](ignored2.py)\n```\n',encoding="utf-8")
            self.assertEqual(list(delivery.markdown_links(root/"review.md")),["fig.png","my guide.md"])

    def test_hash_tampering_fails_and_missing_private_source_is_disclosed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root/"evidence.json"
            source.write_text('{"value":1}',encoding="utf-8")
            facts = {"sources_sha256":{"evidence.json":delivery.sha(source),"private.json":"a"*64}}
            result = delivery.check_sources(root,facts)
            self.assertFalse(result["fact_recollection_performed"])
            self.assertEqual(result["unavailable"],["private.json"])
            source.write_text('{"value":2}',encoding="utf-8")
            self.assertIn("hash mismatch",delivery.check_sources(root,facts)["errors"][0])

    def test_live_promotion_null_coercion_and_wrong_denominators_are_rejected(self):
        for group,field,value in (("experiment","challenger_promoted",True),
                                  ("unknown_live_results","profit",0),
                                  ("experiment","full_eligible_evaluation_rows",12614),
                                  ("dataset","id_is_model_feature",True)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.facts)
                changed[group][field] = value
                with self.assertRaises(ValueError): delivery.check_facts(changed)

    def test_numeric_document_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for document in delivery.CLAIM_DOCS:
                target = root/document
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(project.ROOT/document,target)
            target = root/"README.en.md"
            target.write_text(target.read_text(encoding="utf-8").replace("0.820395","0.990000"),encoding="utf-8")
            self.assertTrue(any("0.820395" in error for error in delivery.check_claims(root,self.facts)))

    def test_frozen_business_reuses_read_only_audits_and_blocks_parent_rebuild(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for stage,manifest in project.BUSINESS_STAGES.items():
                path = root/"artifacts"/stage/manifest
                path.parent.mkdir(parents=True)
                path.write_text("{}")
            plan = project.reproduction_plan("business",root)
            self.assertFalse(plan["blocked_reasons"])
            self.assertEqual(len(plan["commands"]),5)
            self.assertTrue(all(command[0] == "scripts/replay_frozen_business.py" for command in plan["commands"]))
            for target in ("data","models"):
                self.assertTrue(project.reproduction_plan(target,root)["blocked_reasons"])

    def test_missing_freeze_with_descendant_or_occupied_run_blocks_build(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertFalse(project.reproduction_plan("business",root)["blocked_reasons"])
            path = root/"artifacts/aa_validation/aa_freeze_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            self.assertTrue(project.reproduction_plan("business",root)["blocked_reasons"])
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root/"reports/sql_business_analysis/summary.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            self.assertTrue(project.reproduction_plan("business",root)["blocked_reasons"])

    def test_existing_dashboard_is_verified_without_regeneration(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root/"dashboard/powerbi/LoanRisk.pbip"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            self.assertEqual(project.reproduction_plan("dashboard",root)["commands"],project.VERIFY["dashboard"])

    def test_dry_run_calls_no_runner_and_real_execution_stops_on_failure(self):
        runner = Mock(side_effect=[SimpleNamespace(returncode=7),SimpleNamespace(returncode=0)])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(project.execute([["first.py"],["second.py"]],dry_run=True,runner=runner),0)
            runner.assert_not_called()
            self.assertEqual(project.execute([["first.py"],["second.py"]],runner=runner),7)
        self.assertEqual(runner.call_count,1)
        self.assertEqual(runner.call_args.kwargs["cwd"],project.ROOT)
        self.assertNotIn("shell",runner.call_args.kwargs)

    def test_read_only_replay_calls_function_not_writer_and_labels_quick_scope(self):
        verifier = SimpleNamespace(verify=Mock(return_value={"status":"passed"}),main=Mock())
        common = SimpleNamespace(paths=Mock(return_value="output"),write_json=Mock())
        result = replay.replay("aa_validation",deep=False,importer=lambda name: verifier if "verify_" in name else common)
        verifier.verify.assert_called_once()
        self.assertFalse(verifier.verify.call_args.kwargs["deep"])
        verifier.main.assert_not_called()
        common.write_json.assert_not_called()
        self.assertTrue(result["read_only"])
        self.assertEqual(result["scope"],"rows_sql_exports_hashes_and_bootstrap")

    def test_blocked_rebuild_returns_failure_without_execution(self):
        with patch("sys.argv",["project.py","reproduce","--target","data"]), \
             patch.object(project,"reproduction_plan",return_value={"commands":[["danger.py"]],"blocked_reasons":["frozen"]}), \
             patch.object(project,"execute") as execute, redirect_stdout(io.StringIO()):
            self.assertEqual(project.main(),2)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
