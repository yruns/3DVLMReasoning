# Convenience targets for the v9 catalog-first work.

.PHONY: v9-check
v9-check:
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
	    -q
	bash scripts/verify_v9_no_dead_refs.sh
	ruff check src/agents src/query_scene src/evaluation/scripts/prepare_pack_qa_inputs.py \
	    src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
	    src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py
