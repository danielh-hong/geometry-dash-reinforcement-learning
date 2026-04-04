from __future__ import annotations

import constants as C

# Exact snapshot provided by user
snapshot = {
    "player_y": 752.0,
    "player_vy": 0.0,
    "on_ground": 1.0,
    "obstacles": [
        0.0, 791.7921445817601, 752.0, 112.0, 112.0,
        0.0, 1122.79214458176, 752.0, 112.0, 112.0,
        0.0, 1625.79214458176, 752.0, 112.0, 112.0,
        0.0, 1952.79214458176, 752.0, 112.0, 112.0,
        1.0, 2437.79214458176, 752.0, 112.0, 112.0,
    ],
}


def parse_obstacles(flat: list[float]) -> list[dict]:
    out = []
    for i in range(0, len(flat), 5):
        otype, x, y, w, h = flat[i : i + 5]
        out.append({
            "kind": "spike" if otype == 0.0 else "block",
            "otype": float(otype),
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
        })
    return out


def merge_adjacent_same_kind(obs: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for o in sorted(obs, key=lambda z: z["x"]):
        if not merged:
            merged.append(o.copy())
            continue
        last = merged[-1]
        contiguous = o["x"] <= (last["x"] + last["w"] + 1.0)
        same_kind = o["kind"] == last["kind"]
        if same_kind and contiguous:
            last["w"] = (o["x"] + o["w"]) - last["x"]
            last["y"] = min(last["y"], o["y"])
            last["h"] = max(last["h"], o["h"])
        else:
            merged.append(o.copy())
    return merged


def explain() -> None:
    player_y = float(snapshot["player_y"])
    player_vy = float(snapshot["player_vy"])
    on_ground = float(snapshot["on_ground"])

    raw = parse_obstacles(snapshot["obstacles"])
    upcoming = merge_adjacent_same_kind(raw)

    print("=" * 96)
    print("RAW SNAPSHOT")
    print("=" * 96)
    print(f"player_y={player_y}, player_vy={player_vy}, on_ground={on_ground}")
    print(f"PLAYER_X={C.PLAYER_X}, GROUND_Y={C.GROUND_Y}, PLAYER_SIZE={C.PLAYER_SIZE}")
    print(f"VISION_LIMIT=784.0, GAME_SPEED={C.GAME_SPEED:.4f}, MAX_FALL_SPEED={C.MAX_FALL_SPEED:.4f}")

    print("\nRAW OBSTACLES (5 slots from debug dict):")
    for idx, o in enumerate(raw):
        print(
            f"  raw[{idx}] kind={o['kind']:5s} otype={o['otype']:.1f} x={o['x']:.6f} y={o['y']:.1f} w={o['w']:.1f} h={o['h']:.1f}"
        )

    print("\nMERGED UPCOMING (what get_normalized_observation uses before truncating to 3):")
    for idx, o in enumerate(upcoming):
        print(
            f"  merged[{idx}] kind={o['kind']:5s} x={o['x']:.6f} y={o['y']:.1f} w={o['w']:.1f} h={o['h']:.1f}"
        )

    vec: list[float] = []
    labels: list[str] = []

    # Player features
    f0 = player_y / C.GROUND_Y
    vec.append(max(0.0, min(10.0, f0)))
    labels.append(f"[0] player_y_norm = {player_y} / {C.GROUND_Y} = {f0:.9f}")

    f1_raw = player_vy / C.MAX_FALL_SPEED
    f1 = max(-1.0, min(1.0, f1_raw))
    vec.append(f1)
    labels.append(f"[1] player_vy_norm = clamp({player_vy} / {C.MAX_FALL_SPEED:.6f}) = {f1:.9f}")

    vec.append(on_ground)
    labels.append(f"[2] on_ground = {on_ground:.1f}")

    # 3 obstacles x 8 features
    max_obstacles = 3
    vision_limit = 784.0
    for i in range(max_obstacles):
        base = 3 + i * 8
        if i < len(upcoming):
            o = upcoming[i]
            rel_x = o["x"] - C.PLAYER_X
            if rel_x > vision_limit:
                for j in range(8):
                    vec.append(0.0)
                    labels.append(
                        f"[{base + j}] masked=0.0 because rel_x={rel_x:.6f} > {vision_limit}"
                    )
                continue

            otype = 0.0 if o["kind"] == "spike" else 1.0
            rel_y = o["y"] - player_y
            time_to_reach = rel_x / C.GAME_SPEED if C.GAME_SPEED > 0 else 0.0
            gap_top = max(0.0, o["y"] - (player_y + C.PLAYER_SIZE))
            gap_bot = (
                max(0.0, (player_y - C.PLAYER_SIZE) - (o["y"] + o["h"]))
                if o["kind"] == "block"
                else 0.0
            )

            vals = [
                otype,
                max(0.0, min(1.0, rel_x / vision_limit)),
                max(-1.0, min(1.0, rel_y / C.GROUND_Y)),
                o["w"] / C.BLOCK_SIZE / 5.0,
                o["h"] / C.BLOCK_SIZE / 5.0,
                max(0.0, min(1.0, time_to_reach / 6.0)),
                max(0.0, min(1.0, gap_top / C.GROUND_Y)),
                max(0.0, min(1.0, gap_bot / C.GROUND_Y)),
            ]
            desc = [
                f"type = {otype:.1f}",
                f"rel_x_norm = ({rel_x:.6f}/{vision_limit})",
                f"rel_y_norm = ({rel_y:.6f}/{C.GROUND_Y})",
                f"w_norm = {o['w']:.1f}/{C.BLOCK_SIZE}/5",
                f"h_norm = {o['h']:.1f}/{C.BLOCK_SIZE}/5",
                f"time_to_reach_norm = (({rel_x:.6f}/{C.GAME_SPEED:.6f})/6)",
                f"gap_top_norm = max(0, {o['y']:.1f} - ({player_y:.1f}+{C.PLAYER_SIZE}))/{C.GROUND_Y}",
                f"gap_bot_norm = {'0 (spike)' if o['kind']=='spike' else 'block formula'}/{C.GROUND_Y}",
            ]
            for j in range(8):
                vec.append(vals[j])
                labels.append(f"[{base + j}] obs{i+1} {desc[j]} = {vals[j]:.9f}")
        else:
            for j in range(8):
                vec.append(0.0)
                labels.append(f"[{base + j}] pad=0.0 (no obstacle {i+1})")

    # duplicate on_ground
    vec.append(on_ground)
    labels.append(f"[27] is_jump_possible (duplicate on_ground) = {on_ground:.1f}")

    print("\n" + "=" * 96)
    print("EXACT NN VECTOR (index-by-index)")
    print("=" * 96)
    for line in labels:
        print(line)

    print("\n" + "=" * 96)
    print("VECTOR SUMMARY")
    print("=" * 96)
    print(f"length={len(vec)}")
    print("vector=")
    print([round(v, 9) for v in vec])

    print("\n" + "=" * 96)
    print("WHY gap_top and gap_bottom are zero in this snapshot")
    print("=" * 96)
    print("- gap_top compares obstacle top vs player bottom: obstacle_y=752, player_bottom=864, so 752-864 < 0 -> 0")
    print("- first 4 visible objects are spikes, and for spikes gap_bottom is forced to 0 by design")
    print("- if you expected one block, that is width/height normalization (112/112/5 = 0.2), not a gap feature")


if __name__ == "__main__":
    explain()
