#!/usr/bin/env python3
"""assemble_pack_1017.py --experts <PixelML 1017 EXL3 3bpw dir> --engram <convert_engram_int4.py output of the 1017 base>
--sidecars <1017 base snapshot (tokenizer, encoding/, inference/)> --config <diffbot config.json> --out <dir>
[--drafter <quantize_mtp_exl3.py output, shards 44-46>]

diffbot-layout pack for DeepSeek-V4.1-Flash-1017 (hardlinks, all inputs on the out filesystem):
  model-00001..00046-of-00048  PixelML shards (EXL3 3.0 bpw MUL1 routed experts, FP8 non-routed, MXFP4 drafter in 44-46)
                               or, with --drafter, 44-46 from the EXL3 3-bit drafter re-encode
  model-00047..00048-of-00048  int4 Engram tables (1017 base shards 47-48 re-encoded by convert_engram_int4.py)
The index is rebuilt from the shard headers (PixelML's index lists .mcg markers that its shards do not contain).
config.json = the diffbot config (architecture fields are identical for 1017); without --drafter its quantization_config
switches the drafter to source-format experts (mtp_experts "source", mtp_experts_start_layer 40)."""
import argparse, json, os, shutil, struct
ap = argparse.ArgumentParser()
for a in ("--experts", "--engram", "--sidecars", "--config", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--drafter")
args = ap.parse_args()
put = lambda s, d: os.link(os.path.realpath(s), d)
def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))
src = []
for n in range(1, 49):
    name = f"model-{n:05d}-of-00048.safetensors"
    if n >= 47:
        src.append(f"{args.engram}/{name}")
    elif n >= 44 and args.drafter:
        src.append(f"{args.drafter}/{name}")
    else:
        src.append(f"{args.experts}/{name}")
missing = [s for s in src if not os.path.exists(s)]
assert not missing, f"missing shards: {missing}"
os.makedirs(args.out, exist_ok=True)
wm, total = {}, 0
for s in src:
    dst = f"{args.out}/{os.path.basename(s)}"
    if not os.path.exists(dst):
        put(s, dst)
    for k, v in header(dst).items():
        if k != "__metadata__" and k not in wm:
            wm[k] = os.path.basename(s)
            total += v["data_offsets"][1] - v["data_offsets"][0]
json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))},
          open(f"{args.out}/model.safetensors.index.json", "w"), indent=2)
cfg = json.load(open(args.config))
q = cfg["quantization_config"]
if args.drafter:
    q["source"] = "PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw (routed experts + non-routed); Engram int4 from deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10; DSpark drafter routed experts EXL3 3bpw MCG (uncalibrated)"
else:
    q["mtp_experts"] = "source"
    q["mtp_experts_start_layer"] = 40
    q["layer_bits"] = {k: v for k, v in q["layer_bits"].items() if int(k) < 40}
    q["source"] = "PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw (routed experts + non-routed, MXFP4 drafter as stored); Engram int4 from deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10"
json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "LICENSE", "encoding", "inference"):
    s, d = f"{args.sidecars}/{f}", f"{args.out}/{f}"
    if os.path.exists(s) and not os.path.exists(d):
        shutil.copytree(s, d, copy_function=put) if os.path.isdir(s) else put(s, d)
exl3 = sum(k.endswith(".trellis") for k in wm)
mtp = sum(k.startswith("mtp.") and k.endswith(".trellis") for k in wm)
print(f"{args.out}: {len(src)} shards, {len(wm)} tensors, {total / 1e9:.1f} GB indexed, {exl3} EXL3 tensors ({mtp} drafter)")
