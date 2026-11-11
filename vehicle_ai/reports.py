"""Xuất CSV/JSON UTF-8 hỗ trợ tiếng Việt và Excel."""
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = ['frame', 'time_seconds', 'track_id', 'loai_xe', 'ma_lop',
          'do_tin_cay_xe', 'mau_xe', 'do_tin_cay_mau', 'cach_lay_mau', 'bien_so',
          'do_tin_cay_bien_so', 'trang_thai_bien_so',
          'x1', 'y1', 'x2', 'y2']


def rows_for(vehicles, frame=0, time_seconds=0):
    return [{'frame': frame, 'time_seconds': round(time_seconds, 3), **v.to_dict()}
            for v in vehicles]


def write_reports(rows: list[dict], prefix: Path, context: dict | None = None):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    context = context or {}
    labels = Counter(r['loai_xe'] for r in rows)
    colors = Counter(r['mau_xe'] for r in rows)
    ids = {r['track_id'] for r in rows if r['track_id'] != ''}
    report = {
        'chu_y': 'Số lần phát hiện tính theo KHUNG HÌNH; không phải tổng xe duy nhất.',
        'so_luot_phat_hien': len(rows),
        'so_id_theo_doi_uoc_luong': len(ids),
        'so_luot_theo_loai': dict(labels), 'so_luot_theo_mau': dict(colors),
        **context,
    }
    with prefix.with_suffix('.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
