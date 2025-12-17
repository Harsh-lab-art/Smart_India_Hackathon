
import os
import sys
import subprocess
from collections import Counter

import numpy as np


# ============================================================
# Utility Helpers
# ============================================================

def check_cmd(cmd: str) -> bool:
    """Return True if a command is available on PATH."""
    from shutil import which
    return which(cmd) is not None


def read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    p = np.bincount(arr, minlength=256) / len(arr)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_strings(data: bytes, min_len: int = 4):
    """Simple printable strings extractor."""
    out = []
    buf = []
    for b in data:
        if 32 <= b < 127:
            buf.append(chr(b))
        else:
            if len(buf) >= min_len:
                out.append("".join(buf))
            buf = []
    if len(buf) >= min_len:
        out.append("".join(buf))
    return out


# ============================================================
# ELF Parsing
# ============================================================

def try_parse_elf(data: bytes):
    """Try parsing ELF header using pyelftools. Returns info dict or None."""
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        return None

    from io import BytesIO
    bio = BytesIO(data)

    try:
        elf = ELFFile(bio)
    except Exception:
        return None

    info = {
        "class": "ELF32" if elf.elfclass == 32 else "ELF64",
        "endian": "little" if elf.little_endian else "big",
        "machine": elf["e_machine"],
    }
    return info


# ============================================================
# Disassembly Plausibility (Capstone)
# ============================================================

def disasm_scores(data: bytes, max_bytes: int = 200_000):
    """
    Try disassembling the blob with multiple architectures using Capstone.

    Returns:
        dict: {arch_name: {"valid_ratio": float, "valid_count": int, "total": int}}
    """
    try:
        from capstone import (
            Cs,
            CS_ARCH_ARM,
            CS_ARCH_ARM64,
            CS_ARCH_MIPS,
            CS_ARCH_X86,
            CS_ARCH_PPC,
            CS_ARCH_RISCV,
            CS_MODE_32,
            CS_MODE_64,
            CS_MODE_LITTLE_ENDIAN,
            CS_MODE_BIG_ENDIAN,
        )
    except Exception:
        # Capstone not available
        return {}

    blob = data[:max_bytes]

    arch_configs = []
    try:
        arch_configs = [
            ("x86",     Cs(CS_ARCH_X86,   CS_MODE_32)),
            ("x86_64",  Cs(CS_ARCH_X86,   CS_MODE_64)),
            ("arm_le",  Cs(CS_ARCH_ARM,   CS_MODE_32 | CS_MODE_LITTLE_ENDIAN)),
            ("arm_be",  Cs(CS_ARCH_ARM,   CS_MODE_32 | CS_MODE_BIG_ENDIAN)),
            ("arm64",   Cs(CS_ARCH_ARM64, CS_MODE_64 | CS_MODE_LITTLE_ENDIAN)),
            ("mips_le", Cs(CS_ARCH_MIPS,  CS_MODE_32 | CS_MODE_LITTLE_ENDIAN)),
            ("mips_be", Cs(CS_ARCH_MIPS,  CS_MODE_32 | CS_MODE_BIG_ENDIAN)),
            ("ppc",     Cs(CS_ARCH_PPC,   CS_MODE_32 | CS_MODE_BIG_ENDIAN)),
            ("riscv",   Cs(CS_ARCH_RISCV, CS_MODE_64 | CS_MODE_LITTLE_ENDIAN)),
        ]
    except Exception:
        # If any Cs() call fails due to mode issues, bail out safely
        return {}

    scores = {}
    for name, cs in arch_configs:
        cs.detail = False
        total = 0
        valid = 0
        try:
            for _ in cs.disasm(blob, 0):
                valid += 1
                total += 1
        except Exception:
            # If disassembly fails for this arch, treat as 0
            pass

        ratio = (valid / total) if total > 0 else 0.0
        scores[name] = {
            "valid_ratio": ratio,
            "valid_count": valid,
            "total": total,
        }

    return scores


# ============================================================
# External Tools: file, binwalk
# ============================================================

def run_file(path: str):
    if not check_cmd("file"):
        return None
    try:
        out = subprocess.check_output(["file", "-b", path])
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None


def run_binwalk(path: str):
    if not check_cmd("binwalk"):
        return []
    try:
        out = subprocess.check_output(["binwalk", path])
        return out.decode("utf-8", errors="ignore").splitlines()
    except Exception:
        return []


# ============================================================
# Protocol State Inference (Static Heuristics)
# ============================================================

def protocol_state_inference(data: bytes, s_list):
    """
    Heuristically infer presence of protocol handling and phases:
      - initialization
      - handshake
      - key exchange
      - encrypted application

    Returns a dict with protocol -> info.
    """
    result = {}

    lower_strings = [x.lower() for x in s_list]

    # ----- TLS / SSL -----
    tls_keywords_handshake = [
        "clienthello", "serverhello", "hellorequest",
        "certificate", "certificaterequest", "certificateverify",
        "serverkeyexchange", "clientkeyexchange",
        "finished", "changecipherspec",
        "tlsv1", "tlsv1.1", "tlsv1.2", "tlsv1.3", "sslv3"
    ]
    tls_keywords_crypto = [
        "aes-", "chacha20", "poly1305", "gcm", "ccm",
        "ecdhe", "dhe-", "rsa-", "hmac", "md5", "sha1", "sha256", "sha384"
    ]

    tls_handshake_hits = [kw for kw in tls_keywords_handshake
                          if any(kw in s for s in lower_strings)]
    tls_crypto_hits = [kw for kw in tls_keywords_crypto
                       if any(kw in s for s in lower_strings)]

    # TLS record header pattern in raw bytes: 0x16 0x03 0x01..0x04 (handshake)
    tls_record_count = 0
    blob = data
    for i in range(len(blob) - 5):
        if blob[i] == 0x16 and blob[i+1] == 0x03 and blob[i+2] in (0x00, 0x01, 0x02, 0x03, 0x04):
            tls_record_count += 1

    if tls_handshake_hits or tls_crypto_hits or tls_record_count:
        result["TLS"] = {
            "handshake_keywords": tls_handshake_hits,
            "crypto_keywords": tls_crypto_hits,
            "record_header_count": tls_record_count,
            "phases": {
                "initialization": bool(tls_handshake_hits or tls_crypto_hits),
                "handshake": bool(tls_handshake_hits or tls_record_count),
                "key_exchange": any(
                    k in tls_crypto_hits
                    for k in ["ecdhe", "dhe-", "rsa-"]
                ) or any(
                    k in tls_handshake_hits
                    for k in ["serverkeyexchange", "clientkeyexchange"]
                ),
                "encrypted_phase": bool(tls_crypto_hits),
            },
        }

    # ----- SSH -----
    ssh_banners = [s for s in s_list if s.startswith("SSH-") or "ssh-" in s.lower()]
    if ssh_banners:
        result["SSH"] = {
            "banners": ssh_banners[:5],
            "phases": {
                "initialization": True,
                "handshake": True,   # SSH always negotiates early
                "key_exchange": True,
                "encrypted_phase": True,
            },
        }

    # ----- IKE / IPsec-like -----
    ike_keywords = ["isakmp", "ikev1", "ikev2", "quick mode", "main mode", "phase 1", "phase 2"]
    ipsec_keywords = ["esp", "ah", "ipsec", "spi "]

    ike_hits = [kw for kw in ike_keywords if any(kw in s for s in lower_strings)]
    ipsec_hits = [kw for kw in ipsec_keywords if any(kw in s for s in lower_strings)]

    if ike_hits or ipsec_hits:
        result["IKE/IPsec"] = {
            "ike_hits": ike_hits,
            "ipsec_hits": ipsec_hits,
            "phases": {
                "initialization": True,
                "handshake": bool(ike_hits),
                "key_exchange": bool(ike_hits),
                "encrypted_phase": bool(ipsec_hits),
            },
        }

    # ----- Generic crypto / key terms (fallback) -----
    generic_keys = ["key schedule", "session key", "pre-shared key", "psk", "nonce", "iv "]
    generic_hits = [kw for kw in generic_keys if any(kw in s for s in lower_strings)]
    if generic_hits and "TLS" not in result and "SSH" not in result and "IKE/IPsec" not in result:
        result["GENERIC_SECURE_PROTO"] = {
            "hits": generic_hits,
            "phases": {
                "initialization": True,
                "handshake": True,
                "key_exchange": True,
                "encrypted_phase": True,
            },
        }

    return result


# ============================================================
# Core Detection Logic
# ============================================================

def detect(fw_path: str) -> Counter:
    """Analyze a single firmware or blob file and return ranking Counter."""
    data = read_file(fw_path)
    entropy = shannon_entropy(data)

    print("\n========================================")
    print(f"Analyzing: {fw_path}")
    print(f"Size: {len(data)} bytes")
    print(f"Entropy: {entropy:.3f}")
    print("========================================")

    rank = Counter()

    # ELF
    elf_info = try_parse_elf(data)
    if elf_info:
        print("\n[+] ELF detected:", elf_info)
        if elf_info.get("machine") is not None:
            rank[f"elf_machine_{elf_info['machine']}"] += 15

    # file(1)
    file_info = run_file(fw_path)
    if file_info:
        print("\n[+] `file` output:", file_info)
        low = file_info.lower()
        if "aarch64" in low:
            rank["arm64"] += 12
        if "arm" in low:
            rank["arm"] += 10
        if "mips" in low:
            rank["mips"] += 10
        if "risc-v" in low or "riscv" in low:
            rank["riscv"] += 10
        if "powerpc" in low or "ppc" in low:
            rank["powerpc"] += 9
        if "x86-64" in low or "x86_64" in low:
            rank["x86_64"] += 9
        if "x86" in low and "64" not in low:
            rank["x86"] += 7

    # Strings / toolchain hints
    s = extract_strings(data)
    num_strings = len(s)
    avg_len = (sum(len(x) for x in s) / num_strings) if s else 0
    has_gcc = any("gcc" in x.lower() for x in s)

    print(f"\n[+] Strings: count={num_strings}, avg_len={avg_len:.1f}, has_gcc={has_gcc}")

    for st in s:
        low = st.lower()
        if "mips" in low:
            rank["mips"] += 4
        if "aarch64" in low:
            rank["arm64"] += 4
        if "arm" in low and "aarch64" not in low:
            rank["arm"] += 4
        if "riscv" in low:
            rank["riscv"] += 4
        if "powerpc" in low or "ppc" in low:
            rank["powerpc"] += 4
        if "x86_64" in low or "x64" in low:
            rank["x86_64"] += 4
        if "i386" in low or "x86" in low:
            rank["x86"] += 3

    # ---- Protocol state inference here ----
    proto_info = protocol_state_inference(data, s)
    if proto_info:
        print("\n[+] Protocol state inference:")
        for proto, info in proto_info.items():
            print(f"    Protocol: {proto}")
            phases = info.get("phases", {})
            print(f"        initialization : {phases.get('initialization', False)}")
            print(f"        handshake      : {phases.get('handshake', False)}")
            print(f"        key_exchange   : {phases.get('key_exchange', False)}")
            print(f"        encrypted_phase: {phases.get('encrypted_phase', False)}")
            # Small boost if secure protocols are present
            if proto in ("TLS", "SSH", "IKE/IPsec"):
                rank[f"has_{proto}"] += 3

    # Binwalk quick look
    bw = run_binwalk(fw_path)
    if bw:
        print("\n[+] Binwalk (first 5 lines):")
        for line in bw[:5]:
            print("    ", line)

    # Disassembly plausibility
    dis = disasm_scores(data)
    if dis:
        print("\n[+] Disassembly plausibility (Capstone):")
        for arch, info in dis.items():
            print(
                f"    {arch:8s} -> "
                f"valid_ratio={info['valid_ratio']:.3f}, "
                f"valid={info['valid_count']}, "
                f"total={info['total']}"
            )
            score = info["valid_ratio"] * (1 + np.log1p(info["valid_count"]))
            rank[arch] += int(score * 10)

    # Final ranking
    print("\n[+] Architecture ranking:")
    if rank:
        for k, v in rank.most_common():
            print(f"    {k:16s}: {v}")
    else:
        print("    (no strong signals; result is inconclusive)")

    return rank


# ============================================================
# Binwalk Extraction & Multi-part Analysis
# ============================================================

def extract_partitions(path: str, outdir: str = "_extracted"):
    """Run binwalk -e and collect extracted files > 1 KB."""
    if not check_cmd("binwalk"):
        print("\n[!] binwalk not installed; skipping extraction.")
        return []

    print(f"\n[+] Running binwalk extraction into '{outdir}' ...")
    os.makedirs(outdir, exist_ok=True)

    try:
        subprocess.run(["binwalk", "-e", "-C", outdir, path], check=False)
    except Exception as e:
        print(f"[!] Error running binwalk: {e}")
        return []

    extracted_files = []
    for root, _, files in os.walk(outdir):
        for f in files:
            full = os.path.join(root, f)
            try:
                if os.path.getsize(full) > 1024:  # ignore tiny files
                    extracted_files.append(full)
            except OSError:
                continue

    print(f"[+] Found {len(extracted_files)} extracted blobs (>1 KB).")
    return extracted_files


def analyze_extracted_parts(parts):
    """Analyze each extracted partition and print a summary."""
    if not parts:
        print("\n[!] No extracted partitions to analyze.")
        return {}

    print("\n=== Analyzing extracted partitions ===")
    all_results = {}

    for p in parts:
        res = detect(p)
        all_results[p] = res

    print("\n=== Summary of partition rankings ===")
    for path, res in all_results.items():
        print(f"\nPartition: {path}")
        for k, v in res.most_common():
            print(f"    {k:16s}: {v}")

    return all_results


# ============================================================
# Entry Point
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python firmware_architecture_detector.py firmware.bin")
        sys.exit(1)

    fw = sys.argv[1]
    if not os.path.isfile(fw):
        print(f"[!] File not found: {fw}")
        sys.exit(1)

    # 1) Analyze original firmware image
    detect(fw)

    # 2) Try binwalk extraction + per-part analysis
    parts = extract_partitions(fw)
    analyze_extracted_parts(parts)


# ========== Entry Point ============
if __name__ == "__main__":
    firmware_path = "D:/Phase_2/P_2_S_2.bin"
    detect(firmware_path)
    detect(sys.argv[1])
