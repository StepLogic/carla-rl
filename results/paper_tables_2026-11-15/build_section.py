#!/usr/bin/env python3
"""Put the generated CARLA tables into the experiments section.

experiments_section_original.tex is the section as the user pasted it on 2026-11-15. This replaces
each CARLA table environment (matched by its \\label) with the generated one from carla_tables.tex,
adds LaTeX comments (invisible in the PDF) above the three sentences the new numbers affect, and
appends the earlier-protocol tables as an appendix block. Nothing else in the text is changed.

Usage: build_section.py experiments_section_original.tex carla_tables.tex experiments_section_updated.tex
"""
import re
import sys

REPLACE = ["tab:junction_nodes", "tab:long_horizon_carla", "tab:map_navigation", "tab:map_navigation_pose"]
APPENDIX = ["tab:map_navigation_orignav", "tab:long_horizon_carla_orignav"]

NOTES = [
    ("We conduct the experiments in Town02 and Town03",
     "% NOTE 2026-11-15 (tables regenerated): the CARLA main table (tab:map_navigation) is now held-out\n"
     "% Town07 for all three policies; Town02 and Town03 appear in tab:junction_nodes, tab:map_navigation_pose\n"
     "% and the appendix tables. This sentence still names Town02 and Town03 only.\n"),
    (r"\textit{\textbf{Q2} -When does a strong low-level policy still need maps?}",
     "% NOTE 2026-11-15: in CARLA under the aligned navigator (Town07, tab:map_navigation and\n"
     "% tab:long_horizon_carla) no hard-band map effect is significant for any policy. The significant\n"
     "% hard-band effect (0.05 -> 0.13, n=300) is in the earlier-protocol pooled table (appendix).\n"),
    ("As stated earlier, gap nodes are intended to provide visual references",
     "% NOTE 2026-11-15: in CARLA the key-only maps localised by TRUE POSE score at or below no map\n"
     "% (tab:map_navigation_pose; the key-only graphs fragment, SCC in its caption). The CARLA evidence is\n"
     "% 'without gap nodes the map does not help', not 'gap nodes close the gap to a pose oracle'.\n"),
]


def table_blocks(tex):
    """label -> text of the table environment, with the comment lines directly above it."""
    out = {}
    for m in re.finditer(r"((?:^%[^\n]*\n)*)(\\begin\{table\}.*?\\end\{table\})", tex, re.S | re.M):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(2))
        if lab:
            out[lab.group(1)] = m.group(1) + m.group(2)
    return out


def main(orig_path, gen_path, out_path):
    orig = open(orig_path).read()
    gen = table_blocks(open(gen_path).read())
    for lab in REPLACE + APPENDIX:
        assert lab in gen, f"generated file lacks {lab}"
    text = orig
    for lab in REPLACE:
        envs = [m for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
                if f"\\label{{{lab}}}" in m.group(0)]
        assert len(envs) == 1, f"{lab}: found {len(envs)} table environments in the original"
        m = envs[0]
        text = text[:m.start()] + gen[lab] + text[m.end():]
    for anchor, note in NOTES:
        assert text.count(anchor) == 1, f"anchor not unique: {anchor}"
        i = text.index(anchor)
        line_start = text.rfind("\n", 0, i) + 1
        text = text[:line_start] + note + text[line_start:]
    text = text.rstrip("\n") + "\n\n" + (
        "% " + "=" * 98 + "\n"
        "% APPENDIX TABLES (new, 2026-11-15): the earlier protocol, kept so the pooled three-town numbers\n"
        "% remain available. Move to the supplementary material, or drop them.\n"
        "% " + "=" * 98 + "\n") + "\n\n".join(gen[lab] for lab in APPENDIX) + "\n"
    open(out_path, "w").write(text)
    print(f"wrote {out_path}: replaced {len(REPLACE)} tables, {len(NOTES)} notes, {len(APPENDIX)} appendix tables")


if __name__ == "__main__":
    main(*sys.argv[1:4])
