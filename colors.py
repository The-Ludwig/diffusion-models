color_sequences_rot = {
    6: {
        "blue":     (87, 144, 252),
        "orange":   (248, 156, 32),
        "red":      (228, 37, 54),
        "purple":   (150, 74, 139),
        "gray":     (156, 156, 161),
        "purple2":  (122, 33, 221),
    },
    8: {
        "blue":       (24, 69, 251),
        "orange":     (255, 94, 2),
        "red":        (201, 31, 22),
        "purple":     (200, 73, 169),
        "gray":       (173, 173, 125),
        "light blue": (134, 200, 221),
        "blue2":      (87, 141, 255),
        "gray2":      (101, 99, 100),
    },
    10: {
        "purple":     (131, 45, 182),
        "orange":     (255, 169, 14),
        "brown":      (169, 107, 89),
        "orange2":    (231, 99, 0),
        "blue":       (63, 144, 218),
        "red":        (189, 31, 1),
        "tan":        (185, 172, 112),
        "gray2":      (113, 117, 129),
        "light blue": (146, 218, 221),
        "gray":       (148, 164, 162),
    },
}

color_sequences = {
    6: {
        "blue":     (87, 144, 252),
        "orange":   (248, 156, 32),
        "red":      (228, 37, 54),
        "purple":   (150, 74, 139),
        "gray":     (156, 156, 161),
        "purple2":  (122, 33, 221),
    },
    8: {
        "blue":       (24, 69, 251),
        "orange":     (255, 94, 2),
        "red":        (201, 31, 22),
        "purple":     (200, 73, 169),
        "gray":       (173, 173, 125),
        "light blue": (134, 200, 221),
        "blue2":      (87, 141, 255),
        "gray2":      (101, 99, 100),
    },
    10: {
        "blue":       (63, 144, 218),
        "orange":     (255, 169, 14),
        "red":        (189, 31, 1),
        "gray":       (148, 164, 162),
        "purple":     (131, 45, 182),
        "brown":      (169, 107, 89),
        "orange2":    (231, 99, 0),
        "tan":        (185, 172, 112),
        "gray2":      (113, 117, 129),
        "light blue": (146, 218, 221),
    },
}


def rgb_to_hex(rgb_triplet):
    r, g, b = rgb_triplet
    channels = (r, g, b)

    return f"{r:02X}{g:02X}{b:02X}"


hex_sequences = {
    n: {name: rgb_to_hex(rgb) for name, rgb in seq.items()}
    for n, seq in color_sequences.items()
}

pref = "axes.prop_cycle: cycler('color', "
post = ")"

print(list(hex_sequences[10].keys()))
print(pref+str(list(hex_sequences[10].values()))+post)

print(list(hex_sequences[8].keys()))
print(pref+str(list(hex_sequences[8].values()))+post)

print(list(hex_sequences[6].keys()))
print(pref+str(list(hex_sequences[6].values()))+post)
