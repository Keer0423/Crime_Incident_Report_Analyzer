"""
run_pipeline.py — Main entry point for the Multimodal Crime / Incident Report Analyzer
3 Modalities: PDF · Audio · Text (NLP)

Usage:
    python run_pipeline.py --demo
    python run_pipeline.py --pdf  data/samples/pdf_samples/HPD_report_burglary.txt
    python run_pipeline.py --audio path/to/call.mp3
    python run_pipeline.py --text  data/samples/text_samples/witness_statement.txt
    python run_pipeline.py --pdf r.pdf --audio a.mp3 --text t.txt
    python run_pipeline.py --pdf r.pdf --audio a.mp3 --text t.txt --output data/outputs
"""

import argparse
import os
import sys

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   🔍  Multimodal Crime / Incident Report Analyzer            ║
║   3 Modalities: PDF · Audio · Text (NLP)                    ║
║   COMP 4XXX — Final Assignment  |  Group of 2               ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Demo records — 3 incidents, 7 records, all 3 modalities represented ───────
# These are used when running:  python run_pipeline.py --demo
# No API key required for demo mode.

DEMO_RECORDS = [
    # ── Incident 1 — Burglary (PDF only) ──────────────────────────────────
    {
        "source_modality": "pdf",
        "source_file": "HPD_Report_20241120_Burglary.pdf",
        "date": "2024-11-20", "time": "09:45",
        "location": "4521 Richmond Ave, Houston, TX 77027",
        "incident_type": "Burglary",
        "status": "under_investigation",
        "description": (
            "Unknown suspect forced entry through rear kitchen window. "
            "Dell laptop, Sony camera, gold necklace, and approx $200 cash stolen. "
            "Partial fingerprint lifted from window sill."
        ),
        "suspects": ["Unknown male, approx. 5'10\", dark clothing, face obscured"],
        "victims": ["Maria Rodriguez, homeowner"],
        "evidence": [
            "Broken window frame with tool marks",
            "Muddy boot prints on kitchen floor (size 10)",
            "Stolen Dell XPS 15 laptop (SN: DX2024-88831)",
            "Partial fingerprint on window sill (sent to forensics)",
        ],
        "officer": "Officer James Kim",
        "confidence_score": 0.93,
        "extraction_method": "text",
    },
    # ── Incident 2 — Assault (PDF + Audio + Text, all three modalities) ───
    {
        "source_modality": "pdf",
        "source_file": "HPD_Report_20241120_Assault.pdf",
        "date": "2024-11-20", "time": "23:00",
        "location": "Westheimer Rd & Montrose Blvd, Houston, TX",
        "incident_type": "Assault",
        "status": "open",
        "description": (
            "Officer report: two males assaulted a victim outside Club Luxe nightclub. "
            "Victim sustained laceration on temple from glass bottle. Suspects fled on foot."
        ),
        "suspects": ["Male in red jacket, approx. 6ft tall"],
        "victims": ["Unknown male, approx. 30s, laceration on temple"],
        "evidence": ["Broken glass bottle (bagged, submitted to lab)", "Blood swab from sidewalk"],
        "officer": "Officer Dana Reeves",
        "confidence_score": 0.91,
        "extraction_method": "text",
    },
    {
        "source_modality": "audio",
        "source_file": "911_call_20241120_2312.mp3",
        "date": "2024-11-20", "time": "23:12",
        "location": "Westheimer Rd & Montrose Blvd, Houston, TX",
        "incident_type": "Assault",
        "status": "open",
        "description": (
            "911 caller reported two men fighting outside Club Luxe. "
            "One suspect struck victim with glass bottle. Victim on ground, bleeding."
        ),
        "suspects": ["Male in red jacket, ~6ft", "Male in white shirt, ~5'8\""],
        "victims": ["Male, 30s, bleeding from temple"],
        "evidence": ["Broken glass bottle on sidewalk"],
        "officer": None,
        "caller_type": "911_caller",
        "urgency": "high",
        "confidence_score": 0.87,
        "transcript": (
            "DISPATCHER: 911, what's your emergency?\n"
            "CALLER: Yes hello — there's a fight outside Club Luxe on Westheimer and Montrose. "
            "Two guys, one of them hit the other with a glass bottle. "
            "There's blood everywhere. Please hurry.\n"
            "DISPATCHER: Are you in a safe location?\n"
            "CALLER: Yes, I'm across the street. The victim is on the ground.\n"
            "DISPATCHER: Units are 2 minutes away. Stay on the line."
        ),
    },
    {
        "source_modality": "text",
        "source_file": "witness_statement_RChen_20241120.txt",
        "date": "2024-11-20", "time": "23:20",
        "location": "Westheimer Rd & Montrose Blvd, Houston, TX",
        "incident_type": "Assault",
        "status": "open",
        "description": (
            "Witness Robert Chen observed two men attack a woman near Club Luxe bus stop, "
            "steal her black leather purse, and flee east toward Montrose parking garage."
        ),
        "suspects": ["Male in red jacket ~6ft", "Male in white hoodie ~5'8\""],
        "victims": ["Woman, mid-30s, black leather purse with gold clasp stolen"],
        "evidence": ["Stolen black leather purse (gold clasp)", "Witness available for lineup"],
        "officer": None,
        "confidence_score": 0.85,
        "urgency": "high",
        "sentiment_tone": "high_distress",
        "sentiment_score": 0.95,
        "ner_persons": ["Robert Chen"],
        "ner_locations": ["Westheimer Road", "Montrose Boulevard", "Club Luxe"],
        "ner_dates": ["November 20th"],
        "topic_scores": {"Assault": 4, "Robbery": 3, "Disturbance": 1},
    },
    # ── Incident 3 — Vandalism (PDF + Audio) ──────────────────────────────
    {
        "source_modality": "pdf",
        "source_file": "HPD_Report_20241119_Vandalism.pdf",
        "date": "2024-11-19", "time": "02:30",
        "location": "500 Crawford St, Houston, TX 77002",
        "incident_type": "Vandalism",
        "status": "open",
        "description": (
            "Graffiti spray-painted on downtown commercial building overnight. "
            "East wall covered in red, black and silver tags. "
            "Damage estimated at $1,200 for removal and repainting."
        ),
        "suspects": ["Unknown — tag signatures suggest 2-3 person crew"],
        "victims": ["Crawford Street Holdings LLC (building owner)"],
        "evidence": ["Spray paint in red, black, silver", "Two distinct tag signatures"],
        "officer": "Officer Priya Nair",
        "confidence_score": 0.88,
        "extraction_method": "ocr",
    },
    {
        "source_modality": "audio",
        "source_file": "dispatch_Unit12_20241119.mp3",
        "date": "2024-11-19", "time": "02:45",
        "location": "500 Crawford St, Houston, TX 77002",
        "incident_type": "Vandalism",
        "status": "open",
        "description": (
            "Officer Nair dispatch call confirming fresh graffiti on east wall of 500 Crawford. "
            "No suspects on scene. Photos being taken for case file."
        ),
        "suspects": [],
        "victims": [],
        "evidence": ["Fresh spray paint (photographs taken)"],
        "officer": "Officer Priya Nair",
        "caller_type": "officer",
        "urgency": "low",
        "confidence_score": 0.82,
        "transcript": (
            "OFFICER NAIR: Dispatch, this is Unit 12 at 500 Crawford. "
            "We have fresh graffiti on the east wall of the building. "
            "Large area, red and black. No suspects on scene. "
            "Photographing now for the case file."
        ),
    },
    # ── Incident 4 — Drug complaint (Text only) ────────────────────────────
    {
        "source_modality": "text",
        "source_file": "tipline_anonymous_20241121.txt",
        "date": "2024-11-21", "time": "14:30",
        "location": "Hermann Park, Houston, TX 77030",
        "incident_type": "Drug",
        "status": "open",
        "description": (
            "Anonymous online tip reporting suspected drug dealing near Hermann Park main fountain. "
            "Grey hoodie male exchanging small plastic bags with multiple individuals including a minor."
        ),
        "suspects": ["Male, approx. 40, grey hoodie, dark jeans"],
        "victims": [],
        "evidence": ["Small plastic bags observed", "Silver Toyota Camry parked on Fannin St"],
        "officer": None,
        "confidence_score": 0.72,
        "urgency": "medium",
        "sentiment_tone": "neutral",
        "sentiment_score": 0.61,
        "ner_persons": [],
        "ner_locations": ["Hermann Park", "Fannin St"],
        "ner_dates": ["November 21"],
        "topic_scores": {"Drug": 5, "Disturbance": 1},
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Crime / Incident Report Analyzer — 3 Modalities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --demo
  python run_pipeline.py --pdf  data/samples/pdf_samples/HPD_report_burglary.txt
  python run_pipeline.py --audio path/to/911_call.mp3
  python run_pipeline.py --text  data/samples/text_samples/witness_statement.txt
  python run_pipeline.py --pdf r.pdf --audio a.mp3 --text t.txt
        """,
    )
    parser.add_argument("--demo",   action="store_true",   help="Run with built-in demo records (no API key needed)")
    parser.add_argument("--pdf",    nargs="+", default=[], help="Path(s) to PDF incident report file(s)")
    parser.add_argument("--audio",  nargs="+", default=[], help="Path(s) to audio file(s) (.mp3 .wav .m4a etc.)")
    parser.add_argument("--text",   nargs="+", default=[], help="Path(s) to text file(s) (.txt .md etc.)")
    parser.add_argument("--output", default="data/outputs", help="Output directory (default: data/outputs)")
    parser.add_argument("--local-whisper", action="store_true",
                        help="Use local Whisper model instead of OpenAI API (audio module)")
    return parser.parse_args()


def run_demo(output_dir: str):
    """Run the full pipeline using hardcoded demo records — no API key required."""
    from src.integrator.merge import run_integration
    print(BANNER)
    print(f"[DEMO] Running with {len(DEMO_RECORDS)} pre-built records across 3 modalities.\n")
    result = run_integration(DEMO_RECORDS, output_dir=output_dir)
    print(f"\n✅ Demo complete — {result['record_count']} fused incident rows written.")
    print(f"   CSV  → {result['paths'].get('latest_csv')}")
    print(f"   JSON → {result['paths'].get('latest_json')}")
    return result


def run_real(pdf_paths, audio_paths, text_paths, output_dir, use_local_whisper):
    """Run the real AI pipeline on user-supplied files."""
    from src.pdf_processor.processor   import process_pdf
    from src.audio_transcriber.transcriber import process_audio
    from src.text_nlp.analyzer         import process_text
    from src.integrator.merge          import run_integration

    print(BANNER)
    records = []
    sep = "─" * 52

    for path in pdf_paths:
        print(f"\n{sep}\n  [PDF] {path}\n{sep}")
        records.append(process_pdf(path))

    for path in audio_paths:
        print(f"\n{sep}\n  [AUDIO] {path}\n{sep}")
        records.append(process_audio(path, use_local_whisper=use_local_whisper))

    for path in text_paths:
        print(f"\n{sep}\n  [TEXT] {path}\n{sep}")
        if os.path.exists(path):
            records.append(process_text(path))
        else:
            records.append(process_text(path, source_label="inline_text.txt"))

    if not records:
        print("No inputs provided. Use --demo or pass files via --pdf / --audio / --text.")
        sys.exit(1)

    print(f"\n{sep}\n  Integration\n{sep}")
    result = run_integration(records, output_dir=output_dir)
    print(f"\n✅ Pipeline complete — {result['record_count']} fused rows written.")
    print(f"   CSV  → {result['paths'].get('latest_csv')}")
    print(f"   JSON → {result['paths'].get('latest_json')}")
    return result


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        run_demo(output_dir=args.output)
    elif args.pdf or args.audio or args.text:
        run_real(
            pdf_paths=args.pdf,
            audio_paths=args.audio,
            text_paths=args.text,
            output_dir=args.output,
            use_local_whisper=getattr(args, "local_whisper", False),
        )
    else:
        print(BANNER)
        print("No inputs given. Use --demo to run a demonstration, or --help for options.")
        sys.exit(0)


if __name__ == "__main__":
    main()
