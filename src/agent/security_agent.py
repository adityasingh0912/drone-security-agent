"""
agent/security_agent.py
Drone Security Analyst Agent using direct Groq API.
No LangChain agents — uses Groq client directly with tool-calling loop.
Compatible with Python 3.13 + latest langchain.
"""

import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.indexer import (
    query_frames_by_object,
    query_frames_by_time,
    query_frames_by_location,
    get_all_alerts,
    get_summary_stats,
)


# ── Tool functions ────────────────────────────────────────────────────────────

def _clean(q: str) -> str:
    return q.strip().strip('\n').strip("'").strip('"')

def search_frames_by_object(keyword: str) -> str:
    rows = query_frames_by_object(_clean(keyword))
    if not rows:
        return f"No frames found containing '{keyword}'."
    return "\n".join(
        f"[Frame {r['frame_id']} | {r['time']} | {r['location']}] {r['description']}"
        for r in rows
    )

def search_frames_by_time(query: str) -> str:
    parts = [p.strip() for p in _clean(query).split(",")]
    start, end = (parts[0], parts[1]) if len(parts) == 2 else ("00:00", "23:59")
    rows = query_frames_by_time(start, end)
    if not rows:
        return f"No frames found between {start} and {end}."
    return "\n".join(
        f"[Frame {r['frame_id']} | {r['time']} | {r['location']}] {r['description']}"
        for r in rows
    )

def search_frames_by_location(location: str) -> str:
    rows = query_frames_by_location(_clean(location))
    if not rows:
        return f"No frames found at '{location}'."
    return "\n".join(
        f"[Frame {r['frame_id']} | {r['time']}] {r['description']}"
        for r in rows
    )

def get_all_alerts_tool(_: str) -> str:
    alerts = get_all_alerts()
    if not alerts:
        return "No alerts triggered."
    return "\n".join(
        f"[{a['severity']}] {a['message']} (Rule: {a['rule_id']})"
        for a in alerts
    )

def get_event_summary(_: str) -> str:
    s = get_summary_stats()
    return (
        f"Total frames: {s['total_frames']}\n"
        f"Total alerts: {s['total_alerts']}\n"
        f"Critical: {s['critical_alerts']}\n"
        f"High: {s['high_alerts']}"
    )

TOOL_MAP = {
    "search_frames_by_object":   search_frames_by_object,
    "search_frames_by_time":     search_frames_by_time,
    "search_frames_by_location": search_frames_by_location,
    "get_all_alerts":            get_all_alerts_tool,
    "get_event_summary":         get_event_summary,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_frames_by_object",
            "description": "Search indexed video frames by detected object or keyword e.g. 'person', 'vehicle', 'bicycle', 'car', 'bus'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Object or keyword to search for"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_frames_by_time",
            "description": "Search frames within a time window. Input format: 'HH:MM,HH:MM'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Time range as 'HH:MM,HH:MM'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_frames_by_location",
            "description": "Search frames by location name e.g. 'Street', 'Main Gate'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name to search"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_alerts",
            "description": "Get all triggered security alerts with severity and rule info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dummy": {"type": "string", "description": "Pass empty string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_summary",
            "description": "Get summary statistics: total frames and alerts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dummy": {"type": "string", "description": "Pass empty string"}
                },
                "required": []
            }
        }
    },
]

SYSTEM_PROMPT = """You are a Drone Security Analyst AI monitoring a property using a docked drone.
You have access to indexed video frames captured by the drone's BLIP vision model.

Your job:
- Analyze security events from indexed frames
- Identify suspicious patterns (loitering, crowd gathering, vehicles)
- Answer questions about specific events with frame IDs and timestamps
- Be concise and professional

Always use tools to look up real data before answering."""


# ── Groq direct client ────────────────────────────────────────────────────────

def run_agent(query: str) -> dict:
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if not groq_key:
        return _mock_agent(query)

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ]
        steps = []

        # Agentic loop — max 5 tool calls
        for _ in range(5):
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.6,
                max_completion_tokens=2048,
            )

            msg = response.choices[0].message

            # No tool call — final answer
            if not msg.tool_calls:
                return {
                    "query":  query,
                    "answer": msg.content or "No answer generated.",
                    "steps":  steps,
                }

            # Execute tool calls
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args    = json.loads(tc.function.arguments)
                except Exception:
                    args    = {}

                # Call the actual tool
                fn      = TOOL_MAP.get(fn_name)
                if fn:
                    arg_val = list(args.values())[0] if args else ""
                    result  = fn(arg_val)
                else:
                    result  = f"Unknown tool: {fn_name}"

                steps.append({
                    "action": fn_name,
                    "input":  args,
                    "output": result,
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        # Fallback after max iterations
        final = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            temperature=0.6,
            max_completion_tokens=1024,
        )
        return {
            "query":  query,
            "answer": final.choices[0].message.content or "Analysis complete.",
            "steps":  steps,
        }

    except Exception as e:
        return {
            "query":  query,
            "answer": f"Agent error: {str(e)}",
            "steps":  [],
        }


def _mock_agent(query: str) -> dict:
    """Fallback when no Groq key — uses direct DB queries."""
    q = query.lower()
    steps = []

    if any(w in q for w in ["person", "people", "pedestrian", "loiter"]):
        result = search_frames_by_object("person")
        steps.append({"action": "search_frames_by_object", "input": "person", "output": result})
        answer = f"Person activity found:\n{result}"

    elif any(w in q for w in ["vehicle", "car", "bus", "bike", "bicycle"]):
        result = search_frames_by_object("vehicle")
        steps.append({"action": "search_frames_by_object", "input": "vehicle", "output": result})
        answer = f"Vehicle activity found:\n{result}"

    elif any(w in q for w in ["alert", "warning", "critical", "high"]):
        result = get_all_alerts_tool("")
        steps.append({"action": "get_all_alerts", "input": "", "output": result})
        answer = f"Security alerts:\n{result}"

    elif any(w in q for w in ["summary", "stat", "total", "how many"]):
        result = get_event_summary("")
        steps.append({"action": "get_event_summary", "input": "", "output": result})
        answer = f"Surveillance summary:\n{result}"

    else:
        result = get_event_summary("")
        steps.append({"action": "get_event_summary", "input": "", "output": result})
        answer = f"No GROQ_API_KEY set. Here is the current data:\n{result}\nSet GROQ_API_KEY in .env for full AI reasoning."

    return {"query": query, "answer": answer, "steps": steps}