"""Writes test/fixtures/edge_replies.json: what the installed `dream` client does with edge replies (npm run edge).

Each case is one gateway reply, fed to `dream.client.PolicyClient.get_action` in memory (the transport is replaced, so
nothing reaches the network), with strict mode on and off and an optional embodiment. The fixture records, per case
and mode, what Python returns (the action dict, numbers encoded so JSON can carry them: `{"$float": "nan" | "inf" |
"-inf" | "-0"}`, `{"$int": "<digits>"}` for an integer a double cannot hold) or the exception it raises, and the
session id it adopts. The TypeScript client and proxy are held to the same outcome (test/edge-replies.test.ts),
except where a case names a deliberate difference from `DELIBERATE`, each with its reason.

    $DREAM_PYTHON -I scripts/gen_edge_replies.py
"""

import json
import math
import os
import pathlib
import sys

os.environ.setdefault("DREAM_LOG", "off")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "test" / "fixtures" / "edge_replies.json"
sys.path.insert(0, str(ROOT / "scripts"))
import manifest  # noqa: E402

# The deliberate differences from Python: each case that differs names one, and the TypeScript outcome it has.
DELIBERATE = {
    "non_finite": "a joint target or any number of an action is never NaN or an infinity: Python returns them (JSON's 1e400, a null sample filtered into NaN)",
    "not_a_number": "a string, boolean or object where a joint value belongs is refused: numpy converts '0.5' and true, and Python returns other leaves of heads it does not filter",
    "inexact_integer": "an integer a double cannot hold is refused before delivery: Python keeps it exact as an int, JavaScript cannot",
    "checked_heads": "per_key's array parts and the embodiment's declared heads are checked as joint values whatever strict says: Python checks them only in strict mode",
    "proxy_limit": "the proxy withholds a session id longer than 4096 characters rather than pass an unbounded value",
    "python_error": "Python fails with numpy's ValueError (not a DreamError); this client refuses the reply as a DreamActionError",
    "raw_unfiltered": "processing 'raw' (TypeScript only) runs no filter: a chunk Python's filter fails on is returned as it came",
}
# What the TypeScript client raises for each deliberate difference.
TS_OUTCOME = {reason: {"proxy_limit": "DreamProtocolError", "raw_unfiltered": "ok"}.get(reason, "DreamActionError") for reason in DELIBERATE}

OK = '{"status":{"code":"DA-0000"},"session_id":"s1","actions":%s}'
ROWS = "[[0.1,0.2,0.3],[0.11,0.21,0.31],[0.12,0.22,0.32],[0.13,0.23,0.33]]"
SPEC_HEAD = {"action": {"head": {"shape": [2, 2]}}}
SPEC_LABEL = {"action": {"head_label": {}, "q_r_chunk": {"shape": [4, 3]}}}
BODY_OBSERVATION = {"state.right_arm": [0.0, 0.0], "video.front": ["Zg=="], "prompt": "x"}
BODY = {
    "state": {"right_arm": [0, 2]},
    "action": {"right_arm": [0, 2], "right_grip": [2, 3]},
    "video": {"front": "observation.images.front"},
}


def case(name, actions=None, *, reply=None, spec=None, observation=None, deliberate=None):
    """One case: the reply (OK with `actions`, or `reply` as written), its embodiment, and its deliberate differences
    per path ('direct', 'proxy', 'raw'), each a DELIBERATE key."""
    return {
        "name": name,
        "reply": reply if reply is not None else OK % actions,
        "spec": spec,
        "observation": observation or {"prompt": "x"},
        "deliberate": deliberate or {},
    }


ALL = ("direct", "proxy", "raw")


def cases():
    out = [case("valid chunk", f'{{"q_r_chunk":{ROWS}}}')]
    leaves = {"string number": '"0.5"', "string word": '"x"', "true": "true", "object": '{"x":0}', "null": "null", "list": "[0]"}
    for label, leaf in leaves.items():
        head = f"[[0.1,0.2],[{leaf},0.3]]"
        reason = "non_finite" if label == "null" else "not_a_number"
        out.append(case(f"q_r_chunk leaf {label}", f'{{"q_r_chunk":{head}}}', deliberate=dict.fromkeys(ALL, reason)))
        out.append(case(f"per_key part leaf {label}", f'{{"per_key":{{"action.right_arm":{head}}}}}', deliberate=dict.fromkeys(ALL, "checked_heads")))
        out.append(case(f"declared head leaf {label}", f'{{"head":{head}}}', spec=SPEC_HEAD, deliberate=dict.fromkeys(ALL, "checked_heads")))
    for label, head in {"q_r_chunk": "q_r_chunk", "q_l_chunk": "q_l_chunk"}.items():
        out.append(case(f"{label} null head", f'{{"{head}":null}}'))
    out.append(case("per_key null part", '{"per_key":{"action.right_arm":null,"action.right_grip":[[0.5],[0.5]]}}'))
    out.append(case("declared head null", '{"head":null}', spec=SPEC_HEAD))
    for label, head in {"row and number": "[[1,2],3]", "short row": "[[1],[2,3]]", "deep ragged": "[[[1]],[[2,3]]]"}.items():
        out.append(case(f"q_r_chunk ragged {label}", f'{{"q_r_chunk":{head}}}'))
        out.append(case(f"per_key ragged {label}", f'{{"per_key":{{"action.right_arm":{head}}}}}', deliberate=dict.fromkeys(ALL, "checked_heads")))
    for label, head in {"number": "5", "flat": "[1,2,3]", "three deep": "[[[1,2]],[[3,4]]]", "empty": "[]", "empty row": "[[]]", "one row": "[[1,2,3]]"}.items():
        out.append(case(f"q_r_chunk bare {label}", f'{{"q_r_chunk":{head}}}', deliberate={"direct": "python_error", "proxy": "python_error", "raw": "raw_unfiltered"} if label == "empty row" else None))
    out.append(case("label at top level", f'{{"q_r_chunk":{ROWS},"head_label":"right"}}'))
    out.append(case("label declared", f'{{"q_r_chunk":{ROWS},"head_label":"right"}}', spec=SPEC_LABEL))
    out.append(case("label inside per_key", '{"per_key":{"action.right_arm":[[1,2]],"label":"right","count":3,"none":null}}'))
    out.append(case("per_key declared dict", '{"per_key":{"action.right_arm":[[1,2],[1,2]],"action.right_grip":[[0.5],[0.5]]}}', spec=BODY, observation=BODY_OBSERVATION))
    out.append(case("per_key declared wrong width", '{"per_key":{"action.right_arm":[[1,2,3],[1,2,3]],"action.right_grip":[[0.5],[0.5]]}}', spec=BODY, observation=BODY_OBSERVATION))
    out.append(case("per_key declared missing part", '{"per_key":{"action.right_arm":[[1,2],[1,2]]}}', spec=BODY, observation=BODY_OBSERVATION))
    out.append(case("per_key declared null part", '{"per_key":{"action.right_arm":null,"action.right_grip":[[0.5],[0.5]]}}', spec=BODY, observation=BODY_OBSERVATION))
    for label, value in {"string": '"hello"', "number": "3.5", "true": "true", "null": "null", "list with null": "[1,null,\"a\"]", "object": '{"a":{"b":[1,2]}}', "empty object": "{}"}.items():
        out.append(case(f"extension {label}", f'{{"q_r_chunk":{ROWS},"extra":{value}}}'))
    out.append(case("extension 1e400", f'{{"q_r_chunk":{ROWS},"extra":[1e400]}}', deliberate=dict.fromkeys(ALL, "non_finite")))
    out.append(case("extension -1e400", '{"extra":-1e400}', deliberate=dict.fromkeys(ALL, "non_finite")))
    out.append(case("extension -0", '{"extra":-0}'))
    out.append(case("extension -0.0", '{"extra":[-0.0,0.0]}'))
    for label, number in {"2^53": "9007199254740992", "2^53+1": "9007199254740993", "2^53+2": "9007199254740994", "1e18": "1000000000000000000", "1e18+128": "1000000000000000128", "1e18+1": "1000000000000000001", "-2^53-1": "-9007199254740993"}.items():
        exact = int(number) == int(float(number))
        out.append(case(f"extension integer {label}", f'{{"extra":{number}}}', deliberate={} if exact else {"direct": "inexact_integer", "proxy": "inexact_integer", "raw": "inexact_integer"}))
        out.append(case(f"per_key integer {label}", f'{{"per_key":{{"action.right_arm":[[{number}]]}}}}', deliberate={} if exact else dict.fromkeys(ALL, "inexact_integer")))
    out.append(case("q_r_chunk 1e400", '{"q_r_chunk":[[0.1,1e400],[0.2,0.3]]}', deliberate=dict.fromkeys(ALL, "non_finite")))
    out.append(case("per_key 1e400", '{"per_key":{"action.right_arm":[[1e400]]}}', deliberate=dict.fromkeys(ALL, "non_finite")))
    out.append(case("q_r_chunk -0", '{"q_r_chunk":[[-0,0.5],[0.5,-0.0]]}'))
    for length in (1, 257, 4096, 4097):
        sid = "s" * length
        out.append(case(f"session id {length}", reply=f'{{"status":{{"code":"DA-0001","reason":"new"}},"session_id":"{sid}","actions":{{"q_r_chunk":{ROWS}}}}}', deliberate={"proxy": "proxy_limit"} if length > 4096 else {}))
    refusals = {
        "actions null": '{"status":{"code":"DA-2000","reason":"pool_full","retryable":true,"retry_after_s":5},"actions":null}',
        "junk actions": '{"status":{"code":"DA-2000","reason":"pool_warming","retryable":true,"retry_after_s":2},"actions":"junk"}',
        "overflowing actions": '{"status":{"code":"DA-2000","reason":"pool_full","retryable":true,"retry_after_s":5},"actions":{"q_r_chunk":[[1e400]]}}',
        "bad request": '{"status":{"code":"DA-4000","reason":"malformed","detail":"x"}}',
        "server fault": '{"status":{"code":"DA-5000","reason":"error","retryable":true}}',
        "auth": '{"status":{"code":"DA-1000","reason":"unknown_key"}}',
        "expired": '{"status":{"code":"DA-0001","reason":"expired"},"session_id":"s2","actions":{}}',
    }
    for label, reply in refusals.items():
        out.append(case(f"refusal {label}", reply=reply))
    timings = {"missing": "", "null": ',"timing":null', "junk": ',"timing":"slow"', "unfamiliar": ',"timing":{"gateway_ms":4,"units":"ms"}', "strings": ',"timing":{"gateway_ms":"4"}'}
    for label, timing in timings.items():
        out.append(case(f"timing {label}", reply=f'{{"status":{{"code":"DA-0000"}},"session_id":"s1","actions":{{"q_r_chunk":{ROWS}}}{timing}}}'))
    for label, actions in {"null": "null", "missing": None, "string": '"text"', "list": "[1,2]", "empty": "{}"}.items():
        reply = '{"status":{"code":"DA-0000"},"session_id":"s1"}' if actions is None else OK % actions
        out.append(case(f"actions {label}", reply=reply))
    for label, per_key in {"string": '"x"', "null": "null", "list": "[1,2]", "number": "4"}.items():
        out.append(case(f"per_key {label}", f'{{"per_key":{per_key}}}'))
    out.append(case("action key __proto__", f'{{"q_r_chunk":{ROWS},"__proto__":{{"polluted":true}}}}'))
    out.append(case("per_key part __proto__", '{"per_key":{"__proto__":[[1,2]],"action.right_arm":[[1,2]]}}'))
    out.append(case("no status", reply='{"session_id":"s1","actions":{}}'))
    # Ten more, tried against the result: each one expected to break it.
    out.append(case("attack q_r_chunk ragged three deep", '{"q_r_chunk":[[[1],[2,3]]]}'))
    out.append(case("attack q_r_chunk underflow 1e-400", '{"q_r_chunk":[[1e-400,0.5],[0.5,0.5]]}'))
    out.append(case("attack per_key part of objects", '{"per_key":{"action.a":[{"x":1}]}}', deliberate=dict.fromkeys(ALL, "checked_heads")))
    out.append(case("attack duplicate head, last a string", '{"q_r_chunk":[[1,2]],"q_r_chunk":"x"}'))
    out.append(case("attack q_r_chunk true", '{"q_r_chunk":true}', deliberate=dict.fromkeys(ALL, "not_a_number")))
    out.append(case("attack q_r_chunk numeric string", '{"q_r_chunk":"0.5"}', deliberate=dict.fromkeys(ALL, "not_a_number")))
    out.append(case("attack q_r_chunk object", '{"q_r_chunk":{}}'))
    out.append(case("attack empty session id", reply=f'{{"status":{{"code":"DA-0001","reason":"new"}},"session_id":"","actions":{{"q_r_chunk":{ROWS}}}}}'))
    out.append(case("attack numeric status code", reply='{"status":{"code":0},"session_id":"s1","actions":{}}'))
    out.append(case("attack per_key flat part with null", '{"per_key":{"action.a":[1,null]}}', deliberate=dict.fromkeys(ALL, "checked_heads")))
    out.append(case("attack escaped head key", f'{{"\\u0071_r_chunk":{ROWS}}}'))
    out.append(case("attack extension -0e5", '{"extra":-0e5}'))
    out.append(case("observation object declared scalar", f'{{"q_r_chunk":{ROWS}}}', spec={"state": {"meta": {"shape": []}}}, observation={"prompt": "x", "state.meta": {"a": 1}}))
    return out


def encode(value):
    """JSON-ready, with the numbers JSON cannot carry spelled out."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "nan"}
        if math.isinf(value):
            return {"$float": "inf" if value > 0 else "-inf"}
        if value == 0 and math.copysign(1, value) < 0:
            return {"$float": "-0"}
        return value
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value if int(float(value)) == value else {"$int": str(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return {"$repr": repr(value)}


def run(spec_case, strict):
    from dream import client as client_module
    from dream import transport

    config = {"checkpoint": "c"}
    if spec_case["spec"] is not None:
        config["embodiment"] = spec_case["spec"]
    pc = client_module.PolicyClient(host="127.0.0.1", port=9, session_config=config, strict=strict)
    reply = spec_case["reply"].encode()

    class Fake:
        last_exchange = transport.Exchange(rtt_ms=10.0, request_bytes=100, response_bytes=len(reply))

        def request(self, payload, headers, **kwargs):
            return transport.decode(reply)

        def close(self):
            pass

    pc._transport = Fake()
    try:
        action = pc.get_action(dict(spec_case["observation"]))
        result = {"action": encode(action)}
    except Exception as error:  # noqa: BLE001 - every outcome is recorded
        result = {"error": type(error).__name__}
        for key in ("code", "reason", "retryable", "retry_after_s"):
            if hasattr(error, key):
                result[key] = encode(getattr(error, key))
    result["session_id_length"] = None if pc._session_id is None else len(pc._session_id)
    return result


def main() -> None:
    out = []
    for spec_case in cases():
        entry = dict(spec_case)
        entry["python"] = {"strict": run(spec_case, True), "loose": run(spec_case, False)}
        # A deliberate difference is kept only where it is one: where Python's outcome is not already the TypeScript one.
        entry["deliberate"] = {
            path: reason
            for path, reason in entry["deliberate"].items()
            if any(result.get("error", "ok") != TS_OUTCOME[reason] for result in entry["python"].values())
        }
        out.append(entry)
    OUT.write_text(json.dumps({"deliberate": DELIBERATE, "ts_outcome": TS_OUTCOME, "cases": out}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}: {len(out)} cases", file=sys.stderr)
    name = OUT.relative_to(manifest.ROOT).as_posix()
    manifest.update("gen_edge_replies", {name: {"sha256": manifest.sha256(OUT), "environment": manifest.environment()}})


if __name__ == "__main__":
    main()
