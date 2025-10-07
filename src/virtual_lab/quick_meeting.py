"""Convenience script for launching Virtual Lab meetings from the CLI.

The goal is to provide a single entry point that wires up `run_meeting`
with sensible defaults, so a newcomer can immediately test the system or
adapt it for their own orchestration pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from virtual_lab.agent import Agent
from virtual_lab.constants import DEFAULT_MODEL
from virtual_lab.run_meeting import run_meeting


def _build_agent(name: str, goal: str, role: str, model: str | None) -> Agent:
    """Helper to create an :class:`Agent` with concise CLI options."""

    return Agent(
        title=name,
        goal=goal or "Contribute constructively to the agenda.",
        expertise=role,
        role=role,
        model=model or DEFAULT_MODEL,
    )


def _parse_args() -> argparse.Namespace:
    """Configure the command line interface."""

    parser = argparse.ArgumentParser(
        description="Run a quick multi-agent Virtual Lab meeting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("agenda", help="Short description of the meeting focus.")
    parser.add_argument(
        "--meeting-type",
        choices=("team", "individual"),
        default="team",
        help="Choose whether to run a team or individual+critic meeting.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of iterative rounds before the final summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("meetings"),
        help="Directory to store transcripts (JSON + Markdown).",
    )
    parser.add_argument(
        "--save-name",
        default="discussion",
        help="Base filename for the persisted transcript.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature passed to each agent model.",
    )
    parser.add_argument(
        "--pubmed",
        action="store_true",
        help="Enable the built-in PubMed search tool.",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Optional background context snippets (repeatable).",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Agenda questions to highlight during the meeting.",
    )
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        help="Explicit guidelines or constraints to keep in mind.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Summaries from prior meetings that should be referenced.",
    )
    parser.add_argument(
        "--lead",
        default="Team Lead",
        help="Name/title for the team lead persona.",
    )
    parser.add_argument(
        "--lead-role",
        default="Principal Investigator",
        help="Short description of the team lead's expertise/role.",
    )
    parser.add_argument(
        "--lead-goal",
        default="Coordinate the team and deliver a final summary.",
        help="Motivation statement for the team lead.",
    )
    parser.add_argument(
        "--lead-model",
        default=None,
        help="Override model name for the team lead (e.g., gpt-4.1).",
    )
    parser.add_argument(
        "--members",
        default="Scientist A,Scientist B",
        help="Comma-separated list of team members (team meetings only).",
    )
    parser.add_argument(
        "--member-role",
        default="Domain Expert",
        help="Role assigned to each non-lead team member.",
    )
    parser.add_argument(
        "--member-goal",
        default="Share insights that advance the agenda.",
        help="Default motivation for team members.",
    )
    parser.add_argument(
        "--member-model",
        default=None,
        help="Override model used by each team member.",
    )
    parser.add_argument(
        "--individual",
        default="Researcher",
        help="Persona name for the individual meeting participant.",
    )
    parser.add_argument(
        "--individual-role",
        default="Scientist",
        help="Expertise label for the individual meeting participant.",
    )
    parser.add_argument(
        "--individual-goal",
        default="Answer the agenda questions thoroughly.",
        help="Motivation for the individual meeting participant.",
    )
    parser.add_argument(
        "--individual-model",
        default=None,
        help="Model override for the individual participant.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for running quick meetings via the CLI."""

    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.meeting_type == "team":
        team_lead = _build_agent(
            name=args.lead,
            goal=args.lead_goal,
            role=args.lead_role,
            model=args.lead_model,
        )
        members = [
            _build_agent(
                name=member.strip(),
                goal=args.member_goal,
                role=args.member_role,
                model=args.member_model,
            )
            for member in args.members.split(",")
            if member.strip()
        ]
        if not members:
            raise SystemExit("Provide at least one team member via --members.")

        run_meeting(
            meeting_type="team",
            agenda=args.agenda,
            save_dir=args.output_dir,
            save_name=args.save_name,
            team_lead=team_lead,
            team_members=tuple(members),
            agenda_questions=tuple(args.question),
            agenda_rules=tuple(args.rule),
            summaries=tuple(args.summary),
            contexts=tuple(args.context),
            num_rounds=args.rounds,
            temperature=args.temperature,
            pubmed_search=args.pubmed,
            return_summary=False,
        )
    else:
        participant = _build_agent(
            name=args.individual,
            goal=args.individual_goal,
            role=args.individual_role,
            model=args.individual_model,
        )

        run_meeting(
            meeting_type="individual",
            agenda=args.agenda,
            save_dir=args.output_dir,
            save_name=args.save_name,
            team_member=participant,
            agenda_questions=tuple(args.question),
            agenda_rules=tuple(args.rule),
            summaries=tuple(args.summary),
            contexts=tuple(args.context),
            num_rounds=args.rounds,
            temperature=args.temperature,
            pubmed_search=args.pubmed,
            return_summary=False,
        )


if __name__ == "__main__":
    main()
