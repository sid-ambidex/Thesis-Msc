import pandas as pd
from pympi.Elan import Eaf

eaf_file = r"Pro.eaf"
alignment_csv = r"aligned_segments.csv"
output_csv = "self_compassion_overlap_analysis.csv"

def check_overlap(eaf_start, eaf_end, seg_start, seg_end):
    return not (seg_end < eaf_start or seg_start > eaf_end)

eaf = Eaf(eaf_file)

annotations = []
for tier in eaf.get_tier_names():
    for start, end, value in eaf.get_annotation_data_for_tier(tier):
        annotations.append({
            "tier": tier,
            "start": start / 1000,  
            "end": end / 1000
        })

annotations_df = pd.DataFrame(annotations)
print(f"Loaded {len(annotations_df)} EAF annotations")


alignment_df = pd.read_csv(alignment_csv)
records = []
for _, row in annotations_df.iterrows():
    eaf_start, eaf_end = row["start"], row["end"]
    matching_segs = alignment_df[
        alignment_df.apply(
            lambda r: check_overlap(eaf_start, eaf_end, r["start"], r["end"]),
            axis=1
        )
    ]
    
    combined_text = " ".join(matching_segs["text"].dropna().unique()) \
        if "text" in matching_segs.columns else " ".join(matching_segs["transcript"].dropna().unique())
    
    speaker_texts = []
    for _, seg in matching_segs.iterrows():
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text") or seg.get("transcript")
        if pd.notnull(text):
            speaker_texts.append(f"{speaker}: {text}")

    combined_speakers = " || ".join(speaker_texts)

    records.append({
        "tier": row["tier"],
        "start": eaf_start,
        "end": eaf_end,
        "combined_transcript": combined_text,
        "who_said_what": combined_speakers
    })


result_df = pd.DataFrame(records)
result_df = result_df.sort_values(by="start").reset_index(drop=True)
result_df.to_csv(output_csv, index=False)

print(result_df.head(10))