"""Standalone meeting runner for coordinating LLM agents.

This script does not depend on the Virtual Lab package.  It provides a
compact yet well-documented entry point for spinning up structured meetings
between multiple agents and (optionally) the user.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from openai import OpenAI


@dataclasses.dataclass
class Agent:
    """Lightweight description of a meeting participant."""

    name: str
    model: str
    instructions: str
    temperature: float = 0.6

    def system_prompt(self) -> str:
        """Compose the system message that gives the agent its persona."""
        return textwrap.dedent(
            f"""
            You are {self.name} participating in a collaborative research meeting.
            Stay within your expertise, cite evidence when possible, and keep
            your messages concise but rigorous.

            Your focus: {self.instructions.strip()}
            """
        ).strip()


class MeetingRunner:
    """Drives a multi-round conversation among agents (and optionally the user)."""

    def __init__(
        self,
        agents: Sequence[Agent],
        rounds: int,
        agenda: str,
        user_role: str | None = None,
        interactive: bool = False,
        output_path: Path | None = None,
    ) -> None:
        if not agents:
            raise ValueError("At least one agent must be provided.")
        self.agents = list(agents)
        self.rounds = rounds
        self.agenda = agenda.strip()
        self.user_role = user_role
        self.interactive = interactive
        self.output_path = output_path
        self.client = OpenAI()
        self.transcript: List[dict[str, str]] = []

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full meeting workflow."""
        self._record("system", f"Agenda:\n{self.agenda}")
        if self.user_role:
            self._record("system", f"User participates as: {self.user_role}")

        for round_index in range(1, self.rounds + 1):
            print(f"\n=== Round {round_index}/{self.rounds} ===")
            if self.interactive:
                self._gather_user_input(round_index)
            for agent in self.agents:
                reply = self._call_agent(agent)
                self._record(agent.name, reply)
                print(f"\n[{agent.name}]\n{reply}\n")

        self._persist_transcript()

    # ------------------------------------------------------------------
    def _call_agent(self, agent: Agent) -> str:
        """Request a response from the given agent via the Responses API."""
        conversation = self._render_transcript()
        prompt = textwrap.dedent(
            f"""
            You are entering round {len(conversation) + 1} of the meeting. Build on
            the conversation so far. Address the agenda and advance the analysis or
            decision making without repeating earlier points.
            """
        ).strip()

        response = self.client.responses.create(
            model=agent.model,
            input=[
                {"role": "system", "content": agent.system_prompt()},
                {"role": "user", "content": conversation},
                {"role": "user", "content": prompt},
            ],
            temperature=agent.temperature,
        )
        return self._extract_text(response.output)

    # ------------------------------------------------------------------
    def _render_transcript(self) -> str:
        """Format prior turns into a readable block for the next agent."""
        if not self.transcript:
            return "(No prior discussion.)"
        return "\n".join(
            f"{entry['speaker']}: {entry['content']}" for entry in self.transcript
        )

    def _gather_user_input(self, round_index: int) -> None:
        """Prompt the user for a contribution that will be shared with agents."""
        print("Enter your update (or leave blank to skip):")
        user_message = input().strip()
        if user_message:
            speaker = self.user_role or "User"
            self._record(speaker, user_message)

    def _record(self, speaker: str, content: str) -> None:
        self.transcript.append({"speaker": speaker, "content": content})

    def _persist_transcript(self) -> None:
        if not self.output_path:
            return
        payload = {
            "created": datetime.utcnow().isoformat() + "Z",
            "agenda": self.agenda,
            "transcript": self.transcript,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Transcript saved to {self.output_path}")

    @staticmethod
    def _extract_text(chunks: Iterable[object]) -> str:
        """Concatenate text from the Responses API output format."""
        fragments: List[str] = []
        for chunk in chunks:
            if getattr(chunk, "type", None) == "output_text":
                fragments.append(chunk.text)
        text = "".join(fragments).strip()
        if not text:
            raise RuntimeError("LLM response did not contain textual output.")
        return text


# ----------------------------------------------------------------------
def parse_agent(spec: str) -> Agent:
    """Parse "name=Lead,model=gpt-4.1,instructions=Focus on alignment"."""
    fields = {}
    for part in spec.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Agent definition requires key=value pairs: {spec!r}"
            )
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    try:
        return Agent(
            name=fields["name"],
            model=fields.get("model", "gpt-4.1-mini"),
            instructions=fields.get(
                "instructions",
                "Offer generalist scientific reasoning and coordinate the plan.",
            ),
            temperature=float(fields.get("temperature", 0.6)),
        )
    except KeyError as err:  # Missing required field
        raise argparse.ArgumentTypeError(f"Missing field in agent spec: {err.args[0]}")


def default_agents() -> List[Agent]:
    """Provide two sensible personas when none are supplied via CLI."""
    return [
        Agent(
            name="Principal Investigator",
            model="gpt-4.1",
            instructions="Synthesize insights, pose hypotheses, and drive decisions.",
            temperature=0.5,
        ),
        Agent(
            name="Scientific Critic",
            model="gpt-4.1-mini",
            instructions="Stress-test proposals, highlight risks, and demand evidence.",
            temperature=0.7,
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agenda", help="Short description of the meeting agenda.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of speaking rounds to run (default: 3).",
    )
    parser.add_argument(
        "--agent",
        dest="agents",
        action="append",
        type=parse_agent,
        help="Define an agent: name=...,model=...,instructions=...,temperature=...",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Allow the user to add input before each round.",
    )
    parser.add_argument(
        "--user-role",
        help="Optional label for the user when contributing interactively.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Path to save a JSON transcript once the meeting finishes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    agents = args.agents or default_agents()
    runner = MeetingRunner(
        agents=agents,
        rounds=args.rounds,
        agenda=args.agenda,
        user_role=args.user_role,
        interactive=args.interactive,
        output_path=args.save,
    )
    runner.run()


if __name__ == "__main__":
    main()
