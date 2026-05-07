"""
Encrypt and decrypt dataset files for the entiweave project.

Encrypted files are stored in data/encrypted/, mirroring the source directory
structure (e.g. data/encrypted/dataset/deepdive.jsonl.enc).

Usage:
    python -m data.util.dataset_package encrypt
    python -m data.util.dataset_package decrypt
    python -m data.util.dataset_package encrypt --password <pw>
    python -m data.util.dataset_package decrypt --password <pw>
"""

import argparse
import base64
import getpass
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

DATA_DIR = Path(__file__).parent.parent
DATASET_DIR = DATA_DIR / "dataset"
EVAL_DIR = DATA_DIR / "eval"
ENCRYPTED_DIR = DATA_DIR / "encrypted"

_SALT_LEN = 16
_PBKDF2_ITERATIONS = 480_000
_ENC_SUFFIX = ".enc"
_SOURCE_DIRS = [DATASET_DIR, EVAL_DIR]


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def _enc_path(src: Path) -> Path:
    """Map a source file path to its encrypted counterpart under ENCRYPTED_DIR."""
    rel = src.relative_to(DATA_DIR)
    return ENCRYPTED_DIR / rel.with_suffix(rel.suffix + _ENC_SUFFIX)


def _src_path(enc: Path) -> Path:
    """Map an encrypted file path back to its source path under DATA_DIR."""
    rel = enc.relative_to(ENCRYPTED_DIR)
    # Strip the .enc suffix that was appended on top of the original suffix
    original_name = rel.name[: -len(_ENC_SUFFIX)]
    return DATA_DIR / rel.parent / original_name


def encrypt_file(src: Path, password: str) -> Path:
    salt = os.urandom(_SALT_LEN)
    key = _derive_key(password, salt)
    token = Fernet(key).encrypt(src.read_bytes())
    dst = _enc_path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(salt + token)
    return dst


def decrypt_file(enc: Path, password: str) -> Path:
    raw = enc.read_bytes()
    salt, token = raw[:_SALT_LEN], raw[_SALT_LEN:]
    key = _derive_key(password, salt)
    try:
        data = Fernet(key).decrypt(token)
    except InvalidToken:
        raise ValueError(f"Wrong password or corrupted file: {enc}")
    dst = _src_path(enc)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return dst


def encrypt_all(password: str) -> None:
    files = [f for d in _SOURCE_DIRS for f in d.rglob("*") if f.is_file()]
    if not files:
        print("No files found to encrypt.")
        return
    for src in sorted(files):
        dst = encrypt_file(src, password)
        print(f"  encrypted: {src.relative_to(DATA_DIR)}  ->  {dst.relative_to(DATA_DIR)}")
    print(f"\nDone. {len(files)} file(s) encrypted into {ENCRYPTED_DIR.relative_to(DATA_DIR)}/")


def decrypt_all(password: str) -> None:
    enc_files = list(ENCRYPTED_DIR.rglob(f"*{_ENC_SUFFIX}"))
    if not enc_files:
        print("No encrypted files found in data/encrypted/.")
        return
    errors: list[tuple[Path, str]] = []
    for enc in sorted(enc_files):
        try:
            dst = decrypt_file(enc, password)
            print(f"  decrypted: {enc.relative_to(DATA_DIR)}  ->  {dst.relative_to(DATA_DIR)}")
        except ValueError as e:
            errors.append((enc, str(e)))
            print(f"  ERROR: {e}")
    if errors:
        print(f"\n{len(errors)} file(s) failed to decrypt.")
        sys.exit(1)
    print(f"\nDone. {len(enc_files)} file(s) decrypted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt or decrypt entiweave datasets.")
    parser.add_argument("action", choices=["encrypt", "decrypt"])
    parser.add_argument(
        "--password",
        help="Encryption password (prompted securely if omitted)",
    )
    args = parser.parse_args()

    password = args.password or getpass.getpass("Dataset password: ")
    if not password:
        print("Error: password must not be empty.", file=sys.stderr)
        sys.exit(1)

    if args.action == "encrypt":
        encrypt_all(password)
    else:
        decrypt_all(password)


if __name__ == "__main__":
    main()
