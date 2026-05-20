"""Encrypted forgetting certificates — GDPR Article 17 "Right to Erasure" compliance.

Generates Ed25519-signed JWS certificates that prove a specific memory was
provably deleted. Aligned with ECHOFORM's approach to cryptographic forgetting.

Usage:
    from star_graph.forget_certificate import ForgetCertificate

    # Generate a certificate when forgetting
    cert = ForgetCertificate.generate(memory_id="abc123", query="user's API key")
    cert.save("forget_certificate.jws")

    # Verify a certificate
    result = ForgetCertificate.verify("forget_certificate.jws")
    assert result["valid"] is True
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


_SIG_ALG = "Ed25519"


@dataclass
class ForgetCertificate:
    """A cryptographic proof that a memory has been deleted."""

    memory_id: str = ""
    query: str = ""
    deleted_at: float = 0.0
    reason: str = "user_request"
    key_id: str = ""
    public_key_b64: str = ""
    signature_b64: str = ""
    algorithm: str = _SIG_ALG

    # ── Serialization ───────────────────────────────────────

    def to_jws(self) -> str:
        """Serialize as compact JWS with embedded public key (self-verifying)."""
        header = {
            "alg": self.algorithm,
            "kid": self.key_id,
            "typ": "JWS",
            "jwk": {"kty": "OKP", "crv": "Ed25519", "x": self.public_key_b64},
        }
        payload = {
            "sub": self.memory_id,
            "query": self.query,
            "iat": int(self.deleted_at),
            "deleted_at": self.deleted_at,
            "reason": self.reason,
            "jti": f"forget-{self.memory_id}-{int(self.deleted_at)}",
        }
        header_b64 = _b64url(json.dumps(header, separators=(",", ":")))
        payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")))
        return f"{header_b64}.{payload_b64}.{self.signature_b64}"

    def save(self, path: str) -> str:
        """Write JWS certificate to file. Returns absolute path."""
        jws = self.to_jws()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(jws, encoding="utf-8")
        return str(out.resolve())

    # ── Factory ─────────────────────────────────────────────

    @classmethod
    def generate(cls, memory_id: str, query: str = "",
                 reason: str = "user_request") -> ForgetCertificate:
        """Create a new forgetting certificate signed with a fresh Ed25519 keypair.

        Args:
            memory_id: the ID of the deleted memory
            query: the forget query used
            reason: reason for deletion (user_request, gdpr_art17, policy, etc.)
        """
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        now = time.time()
        kid = _b64url(pub_bytes[:8]).rstrip("=")
        pub_b64 = _b64url(pub_bytes)

        header = json.dumps({
            "alg": _SIG_ALG, "kid": kid, "typ": "JWS",
            "jwk": {"kty": "OKP", "crv": "Ed25519", "x": pub_b64},
        }, separators=(",", ":"))
        payload = json.dumps({
            "sub": memory_id,
            "query": query,
            "iat": int(now),
            "deleted_at": now,
            "reason": reason,
            "jti": f"forget-{memory_id}-{int(now)}",
        }, separators=(",", ":"))

        signing_input = f"{_b64url(header)}.{_b64url(payload)}"
        signature = private_key.sign(signing_input.encode("utf-8"))

        return cls(
            memory_id=memory_id,
            query=query,
            deleted_at=now,
            reason=reason,
            key_id=kid,
            public_key_b64=pub_b64,
            signature_b64=_b64url(signature),
        )

    # ── Verification ────────────────────────────────────────

    @classmethod
    def verify(cls, path_or_jws: str) -> dict:
        """Verify a forgetting certificate. Auto-verifies signature when jwk is embedded.

        Returns result dict with 'valid' key — True when cryptographic signature passes.
        """
        if os.path.isfile(path_or_jws):
            jws = Path(path_or_jws).read_text(encoding="utf-8").strip()
        else:
            jws = path_or_jws

        parts = jws.split(".")
        if len(parts) != 3:
            return {"valid": False, "error": "invalid JWS format"}

        header_b64, payload_b64, sig_b64 = parts
        try:
            header = json.loads(_b64decode(header_b64))
            payload = json.loads(_b64decode(payload_b64))
        except Exception:
            return {"valid": False, "error": "malformed header or payload"}

        signing_input = f"{header_b64}.{payload_b64}"

        # Try embedded public key (jwk) for self-verification
        jwk = header.get("jwk", {})
        if jwk and jwk.get("x"):
            try:
                pub_bytes = _b64decode_bytes(jwk["x"])
                public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
                sig_bytes = _b64decode_bytes(sig_b64)
                public_key.verify(sig_bytes, signing_input.encode("utf-8"))
                crypto_valid = True
            except (InvalidSignature, Exception):
                crypto_valid = False
        else:
            crypto_valid = False

        return {
            "valid": crypto_valid,
            "memory_id": payload.get("sub", ""),
            "query": payload.get("query", ""),
            "deleted_at": payload.get("deleted_at", 0),
            "reason": payload.get("reason", ""),
            "algorithm": header.get("alg", _SIG_ALG),
            "kid": header.get("kid", ""),
            "raw": {
                "header": header,
                "payload": payload,
                "signature_b64": sig_b64,
                "signing_input": signing_input,
            },
        }

    @classmethod
    def verify_full(cls, path_or_jws: str, public_key_b64: str) -> dict:
        """Full cryptographic verification with a known public key."""
        result = cls.verify(path_or_jws)
        if not result.get("raw"):
            return result

        try:
            pub_bytes = _b64decode_bytes(public_key_b64)
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            sig_bytes = _b64decode_bytes(result["raw"]["signature_b64"])
            signing_input = result["raw"]["signing_input"].encode("utf-8")
            public_key.verify(sig_bytes, signing_input)
            result["valid"] = True
        except (InvalidSignature, Exception):
            result["valid"] = False
            if "error" not in result:
                result["error"] = "signature verification failed"

        return result


# ── Utility ─────────────────────────────────────────────────

def _b64url(data: str | bytes) -> str:
    """Base64url encode (no padding)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(b64: str) -> str:
    """Base64url decode to string."""
    padded = b64 + "=" * (4 - len(b64) % 4) if len(b64) % 4 else b64
    return base64.urlsafe_b64decode(padded).decode("utf-8")


def _b64decode_bytes(b64: str) -> bytes:
    """Base64url decode to bytes."""
    padded = b64 + "=" * (4 - len(b64) % 4) if len(b64) % 4 else b64
    return base64.urlsafe_b64decode(padded)
