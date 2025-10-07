"""Runs a meeting with LLM agents."""

import time
from pathlib import Path
from typing import Literal

from openai import OpenAI
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from tqdm import trange, tqdm

from virtual_lab.agent import Agent
from virtual_lab.constants import CONSISTENT_TEMPERATURE, PUBMED_TOOL_DESCRIPTION
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.utils import (
    count_discussion_tokens,
    count_tokens,
    get_summary,
    print_cost_and_time,
    run_tools,
    save_meeting,
)


def _build_input_messages(agent: Agent, discussion: list[dict[str, str]]) -> list[dict]:
    """Construct the request payload for the Responses API."""

    input_messages: list[dict] = [
        {
            "role": "system",
            "type": "message",
            "content": [{"type": "text", "text": agent.prompt}],
        }
    ]

    for turn in discussion:
        role = "assistant" if turn["agent"] == agent.title else "user"
        input_messages.append(
            {
                "role": role,
                "type": "message",
                "content": [{"type": "text", "text": turn["message"]}],
            }
        )

    return input_messages


def _extract_response_text(response_output: list | None) -> str:
    """Extract assistant text from a Responses API payload."""

    if response_output is None:
        return ""

    texts: list[str] = []
    for item in response_output:
        if isinstance(item, ResponseOutputMessage):
            for content in item.content:
                if isinstance(content, ResponseOutputText):
                    text = content.text.strip()
                    if text:
                        texts.append(text)

    return "\n\n".join(texts)


def _collect_function_calls(
    response_output: list | None,
) -> list[ResponseFunctionToolCall]:
    """Return function tool calls found in a response output payload."""

    if response_output is None:
        return []

    return [
        item
        for item in response_output
        if isinstance(item, ResponseFunctionToolCall)
    ]


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search: bool = False,
    return_summary: bool = False,
) -> str:
    """Runs a meeting with a LLM agents.

    :param meeting_type: The type of meeting.
    :param agenda: The agenda for the meeting.
    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param team_lead: The team lead for a team meeting (None for individual meeting).
    :param team_members: The team members for a team meeting (None for individual meeting).
    :param team_member: The team member for an individual meeting (None for team meeting).
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the meeting.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion.
    :param temperature: The sampling temperature.
    :param pubmed_search: Whether to include a PubMed search tool.
    :param return_summary: Whether to return the summary of the meeting.
    :return: The summary of the meeting (i.e., the last message) if return_summary is True, else None.
    """
    # Validate meeting type
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError(
                "Individual meeting does not require team lead or team members"
            )
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    # Start timing the meeting
    start_time = time.time()

    # Set up client
    client = OpenAI()

    # Set up team
    if meeting_type == "team":
        team = [team_lead] + list(team_members)
    else:
        team = [team_member] + [SCIENTIFIC_CRITIC]

    # Prepare tool definitions
    tools = [PUBMED_TOOL_DESCRIPTION] if pubmed_search else None

    # Track running discussion locally
    discussion: list[dict[str, str]] = []

    # Set up tool token count
    tool_token_count = 0

    # Initial prompt for team meeting
    if meeting_type == "team":
        discussion.append(
            {
                "agent": "User",
                "message": team_meeting_start_prompt(
                    team_lead=team_lead,
                    team_members=team_members,
                    agenda=agenda,
                    agenda_questions=agenda_questions,
                    agenda_rules=agenda_rules,
                    summaries=summaries,
                    contexts=contexts,
                    num_rounds=num_rounds,
                ),
            }
        )

    # Token accounting (input/output/max tracked separately from tools)
    token_counts = {"input": 0, "output": 0, "max": 0}

    # Loop through rounds
    for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
        round_num = round_index + 1

        # Loop through team and elicit responses
        for agent in tqdm(team, desc="Team"):
            # Prompt based on agent and round number
            if meeting_type == "team":
                # Team meeting prompts
                if agent == team_lead:
                    if round_index == 0:
                        prompt = team_meeting_team_lead_initial_prompt(
                            team_lead=team_lead
                        )
                    elif round_index == num_rounds:
                        prompt = team_meeting_team_lead_final_prompt(
                            team_lead=team_lead,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                        )
                    else:
                        prompt = team_meeting_team_lead_intermediate_prompt(
                            team_lead=team_lead,
                            round_num=round_num - 1,
                            num_rounds=num_rounds,
                        )
                else:
                    prompt = team_meeting_team_member_prompt(
                        team_member=agent, round_num=round_num, num_rounds=num_rounds
                    )
            else:
                # Individual meeting prompts
                if agent == SCIENTIFIC_CRITIC:
                    prompt = individual_meeting_critic_prompt(
                        critic=SCIENTIFIC_CRITIC, agent=team_member
                    )
                else:
                    if round_index == 0:
                        prompt = individual_meeting_start_prompt(
                            team_member=team_member,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                            summaries=summaries,
                            contexts=contexts,
                        )
                    else:
                        prompt = individual_meeting_agent_prompt(
                            critic=SCIENTIFIC_CRITIC, agent=team_member
                        )

            # Add orchestrator prompt to running discussion
            discussion.append({"agent": "User", "message": prompt})

            # Build request payload for the current agent
            input_messages = _build_input_messages(agent=agent, discussion=discussion)

            # Run the agent with the Responses API
            response_kwargs = {
                "model": agent.model,
                "input": input_messages,
                "temperature": temperature,
            }
            if tools is not None:
                response_kwargs["tools"] = tools

            response = client.responses.create(**response_kwargs)

            usage = response.usage
            if usage is not None:
                token_counts["input"] += usage.input_tokens
                token_counts["output"] += usage.output_tokens
                token_counts["max"] = max(token_counts["max"], usage.total_tokens)

            # Handle any function tool calls
            while True:
                tool_calls = _collect_function_calls(response.output)
                if not tool_calls:
                    break

                tool_outputs = run_tools(tool_calls=tool_calls)

                # Update tool token count
                tool_token_count += sum(
                    count_tokens(tool_output["output"]) for tool_output in tool_outputs
                )

                # Surface tool outputs in the visible discussion
                discussion.append(
                    {
                        "agent": "User",
                        "message": "Tool Output:\n\n"
                        + "\n\n".join(
                            tool_output["output"] for tool_output in tool_outputs
                        ),
                    }
                )

                # Provide tool outputs back to the model
                response = client.responses.create(
                    model=agent.model,
                    previous_response_id=response.id,
                    input=[
                        {
                            "type": "function_call_output",
                            "call_id": tool_output["call_id"],
                            "output": tool_output["output"],
                        }
                        for tool_output in tool_outputs
                    ],
                )

                usage = response.usage
                if usage is not None:
                    token_counts["input"] += usage.input_tokens
                    token_counts["output"] += usage.output_tokens
                    token_counts["max"] = max(
                        token_counts["max"], usage.total_tokens
                    )

            if response.status != "completed":
                raise ValueError(f"Response failed: {response.status}")

            response_text = _extract_response_text(response.output)
            if not response_text:
                raise ValueError("Model returned an empty response")

            discussion.append({"agent": agent.title, "message": response_text})

            # If final round, only team lead or team member responds
            if round_index == num_rounds:
                break

    # Fallback to heuristic token counting if usage details were unavailable
    if token_counts["input"] == 0 and token_counts["output"] == 0:
        token_counts = count_discussion_tokens(discussion=discussion)
    else:
        heuristic_counts = count_discussion_tokens(discussion=discussion)
        token_counts["max"] = max(token_counts["max"], heuristic_counts["max"])

    # Add tool token count to total token count
    token_counts["tool"] = tool_token_count

    # Print cost and time
    # TODO: handle different models for different agents
    print_cost_and_time(
        token_counts=token_counts,
        model=team_lead.model if meeting_type == "team" else team_member.model,
        elapsed_time=time.time() - start_time,
    )

    # Save the discussion as JSON and Markdown
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )

    # Optionally, return summary
    if return_summary:
        return get_summary(discussion)
