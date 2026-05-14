# Convenience targets for the v9 catalog-first work.

.PHONY: v9-check
v9-check:
	# Tests that require optional deps unavailable on the Mac development
	# environment (open3d, torch) are deliberately skipped here. The CI
	# runner that ships these wheels should drop the --ignore flags.
	PYTHONPATH=src .venv/bin/python -m pytest \
	    src/agents/catalog/tests \
	    src/agents/runtime/tests \
	    src/agents/tools/tests \
	    src/agents/skills/tests \
	    src/agents/packs/vg_embodiedscan/tests \
	    src/agents/packs/qa_default/tests \
	    src/agents/packs/qa_default/skills/tests \
	    src/agents/packs/vg_embodiedscan/skills/tests \
	    src/agents/tests \
	    src/agents/core/tests \
	    src/query_scene/tests \
	    src/evaluation/scripts/tests \
	    --ignore=src/agents/packs/vg_embodiedscan/tests/test_clip_provider.py \
	    --ignore=src/query_scene/tests/test_bev_builder.py \
	    --ignore=src/query_scene/tests/test_scene_visualizer.py \
	    --ignore=src/query_scene/tests/test_frustum.py \
	    -q
	bash scripts/verify_v9_no_dead_refs.sh
	.venv/bin/python -m ruff check --select F \
	    src/agents \
	    src/query_scene/scene_bev_builder.py \
	    src/evaluation/scripts/prepare_pack_qa_inputs.py \
	    src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
	    src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py
