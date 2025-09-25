"""
Minimal TFRecord and SequenceExample reader tailored for AudioSet frame-level
features (VGGish 128-dim 8-bit embeddings), without TensorFlow.

This reader implements:
- TFRecord framing (reads records; skips CRC checks)
- Protobuf wire-format parsing for the subset of messages used by
  tensorflow.core.example.SequenceExample:
  - SequenceExample { Features context = 1; FeatureLists feature_lists = 2; }
  - Features { map<string, Feature> feature = 1; }
  - Feature { oneof kind { BytesList bytes_list = 1; FloatList float_list = 2; Int64List int64_list = 3; } }
  - BytesList { repeated bytes value = 1; }
  - FloatList { repeated float value = 1; }  # packed or unpacked
  - Int64List { repeated int64 value = 1; }  # packed or unpacked
  - FeatureLists { map<string, FeatureList> feature_list = 1; }
  - FeatureList { repeated Feature feature = 1; }

Only the necessary subset is decoded. Unknown fields are skipped.

Returned structure for each record:
{
  'context': {
      'video_id': str,
      'start_time_seconds': float,
      'end_time_seconds': float,
      'labels': List[int],
  },
  'feature_lists': {
      'audio_embedding': List[bytes],  # one 128-byte vector per second
  }
}
"""
from __future__ import annotations

import io
import os
import struct
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# ---------------------------
# TFRecord framing
# ---------------------------

def iter_tfrecord_records(path: str) -> Iterator[bytes]:
    """Yield serialized protobuf messages from a TFRecord file.

    This ignores CRC checks for simplicity and speed.
    """
    with open(path, "rb") as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes:
                return
            if len(length_bytes) < 8:
                # Truncated file
                return
            (length,) = struct.unpack("<Q", length_bytes)
            # skip crc of length
            _ = f.read(4)
            data = f.read(length)
            if len(data) < length:
                return
            # skip crc of data
            _ = f.read(4)
            yield data


# ---------------------------
# Protobuf wire format utils
# ---------------------------

def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated varint")
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift >= 64:
            raise ValueError("Varint too long")
    return value, pos


def _skip_field(data: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:  # varint
        _, pos = _read_varint(data, pos)
        return pos
    elif wire_type == 1:  # 64-bit
        return pos + 8
    elif wire_type == 2:  # length-delimited
        length, pos = _read_varint(data, pos)
        return pos + length
    elif wire_type == 5:  # 32-bit
        return pos + 4
    else:
        raise ValueError(f"Unsupported wire type: {wire_type}")


def _read_length_delimited(data: bytes, pos: int) -> Tuple[bytes, int]:
    length, pos = _read_varint(data, pos)
    end = pos + length
    return data[pos:end], end


# ---------------------------
# Example/SequenceExample subset decoders
# ---------------------------

def _parse_bytes_list(data: bytes) -> List[bytes]:
    pos = 0
    values: List[bytes] = []
    n = len(data)
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num == 1:  # value
            if wire_type != 2:
                # Unexpected; skip
                pos = _skip_field(data, pos, wire_type)
                continue
            b, pos = _read_length_delimited(data, pos)
            values.append(b)
        else:
            pos = _skip_field(data, pos, wire_type)
    return values


def _parse_float_list(data: bytes) -> List[float]:
    pos = 0
    n = len(data)
    values: List[float] = []
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num != 1:
            pos = _skip_field(data, pos, wire_type)
            continue
        if wire_type == 5:  # single float (unpacked)
            (val,) = struct.unpack('<f', data[pos:pos+4])
            values.append(val)
            pos += 4
        elif wire_type == 2:  # packed floats
            chunk, pos = _read_length_delimited(data, pos)
            # floats are little-endian 4-byte
            for i in range(0, len(chunk), 4):
                if i + 4 <= len(chunk):
                    (val,) = struct.unpack('<f', chunk[i:i+4])
                    values.append(val)
        else:
            pos = _skip_field(data, pos, wire_type)
    return values


def _parse_int64_list(data: bytes) -> List[int]:
    pos = 0
    n = len(data)
    values: List[int] = []
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num != 1:
            pos = _skip_field(data, pos, wire_type)
            continue
        if wire_type == 0:  # varint (unpacked)
            v, pos = _read_varint(data, pos)
            values.append(v)
        elif wire_type == 2:  # packed varints
            chunk, pos = _read_length_delimited(data, pos)
            cpos = 0
            while cpos < len(chunk):
                v, cpos2 = _read_varint(chunk, cpos)
                values.append(v)
                cpos = cpos2
        else:
            pos = _skip_field(data, pos, wire_type)
    return values


def _parse_feature(data: bytes) -> Dict[str, List]:
    # Returns dict with at most one of: 'bytes_list', 'float_list', 'int64_list'
    pos = 0
    n = len(data)
    result: Dict[str, List] = {}
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type != 2:
            # Feature lists are messages; anything else skip
            pos = _skip_field(data, pos, wire_type)
            continue
        sub, pos = _read_length_delimited(data, pos)
        if field_num == 1:
            result['bytes_list'] = _parse_bytes_list(sub)
        elif field_num == 2:
            result['float_list'] = _parse_float_list(sub)
        elif field_num == 3:
            result['int64_list'] = _parse_int64_list(sub)
        # Unknown fields ignored
    return result


def _parse_features(data: bytes) -> Dict[str, Dict[str, List]]:
    # map<string, Feature> feature = 1;
    pos = 0
    n = len(data)
    features: Dict[str, Dict[str, List]] = {}
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num != 1 or wire_type != 2:
            pos = _skip_field(data, pos, wire_type)
            continue
        entry_bytes, pos = _read_length_delimited(data, pos)
        # Entry: { string key = 1; Feature value = 2; }
        epos = 0
        key: Optional[str] = None
        val: Optional[Dict[str, List]] = None
        while epos < len(entry_bytes):
            etag, epos = _read_varint(entry_bytes, epos)
            efield = etag >> 3
            ewire = etag & 0x7
            if efield == 1 and ewire == 2:
                kbytes, epos = _read_length_delimited(entry_bytes, epos)
                try:
                    key = kbytes.decode('utf-8')
                except Exception:
                    key = kbytes.decode('latin-1')
            elif efield == 2 and ewire == 2:
                vbytes, epos = _read_length_delimited(entry_bytes, epos)
                val = _parse_feature(vbytes)
            else:
                epos = _skip_field(entry_bytes, epos, ewire)
        if key is not None and val is not None:
            features[key] = val
    return features


def _parse_feature_list(data: bytes) -> List[Dict[str, List]]:
    # repeated Feature feature = 1;
    pos = 0
    n = len(data)
    out: List[Dict[str, List]] = []
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num == 1 and wire_type == 2:
            fbytes, pos = _read_length_delimited(data, pos)
            out.append(_parse_feature(fbytes))
        else:
            pos = _skip_field(data, pos, wire_type)
    return out


def _parse_feature_lists(data: bytes) -> Dict[str, List[Dict[str, List]]]:
    # map<string, FeatureList> feature_list = 1;
    pos = 0
    n = len(data)
    out: Dict[str, List[Dict[str, List]]] = {}
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num != 1 or wire_type != 2:
            pos = _skip_field(data, pos, wire_type)
            continue
        entry_bytes, pos = _read_length_delimited(data, pos)
        # Entry: { string key = 1; FeatureList value = 2; }
        epos = 0
        key: Optional[str] = None
        val_list: Optional[List[Dict[str, List]]] = None
        while epos < len(entry_bytes):
            etag, epos = _read_varint(entry_bytes, epos)
            efield = etag >> 3
            ewire = etag & 0x7
            if efield == 1 and ewire == 2:
                kbytes, epos = _read_length_delimited(entry_bytes, epos)
                try:
                    key = kbytes.decode('utf-8')
                except Exception:
                    key = kbytes.decode('latin-1')
            elif efield == 2 and ewire == 2:
                vbytes, epos = _read_length_delimited(entry_bytes, epos)
                val_list = _parse_feature_list(vbytes)
            else:
                epos = _skip_field(entry_bytes, epos, ewire)
        if key is not None and val_list is not None:
            out[key] = val_list
    return out


def parse_sequence_example(data: bytes) -> Dict:
    pos = 0
    n = len(data)
    context: Dict[str, Dict[str, List]] = {}
    feature_lists: Dict[str, List[Dict[str, List]]] = {}
    while pos < n:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type != 2:
            pos = _skip_field(data, pos, wire_type)
            continue
        sub, pos = _read_length_delimited(data, pos)
        if field_num == 1:  # context
            context = _parse_features(sub)
        elif field_num == 2:  # feature_lists
            feature_lists = _parse_feature_lists(sub)
        else:
            # Unknown top-level field
            pass

    # Project to a compact structure
    out_context: Dict[str, object] = {}
    # video_id
    if 'video_id' in context and 'bytes_list' in context['video_id']:
        bl = context['video_id']['bytes_list']
        if bl:
            try:
                out_context['video_id'] = bl[0].decode('utf-8')
            except Exception:
                out_context['video_id'] = bl[0].decode('latin-1')
    # start_time_seconds / end_time_seconds
    if 'start_time_seconds' in context and 'float_list' in context['start_time_seconds']:
        fl = context['start_time_seconds']['float_list']
        if fl:
            out_context['start_time_seconds'] = float(fl[0])
    if 'end_time_seconds' in context and 'float_list' in context['end_time_seconds']:
        fl = context['end_time_seconds']['float_list']
        if fl:
            out_context['end_time_seconds'] = float(fl[0])
    # labels
    if 'labels' in context and 'int64_list' in context['labels']:
        out_context['labels'] = [int(v) for v in context['labels']['int64_list']]

    # audio_embedding feature list -> list of 128-byte values
    audio_embeddings: List[bytes] = []
    if 'audio_embedding' in feature_lists:
        for feat in feature_lists['audio_embedding']:
            if 'bytes_list' in feat and feat['bytes_list']:
                # Expect exactly one bytes value per second
                audio_embeddings.append(feat['bytes_list'][0])

    return {
        'context': out_context,
        'feature_lists': {
            'audio_embedding': audio_embeddings
        }
    }


__all__ = [
    'iter_tfrecord_records',
    'parse_sequence_example',
]

