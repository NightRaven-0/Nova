# brain/tools.py
# Backward-compatible shim. Skills now live in the top-level `skills/` package —
# to add a capability, drop a new module in skills/ and decorate a function with
# @skill(...). It is auto-discovered; no edits here are needed.

from skills import dispatch, get_tool_schemas

# Schemas for every registered skill (the brain passes this as `tools=`).
TOOLS = get_tool_schemas()


def dispatch_tool(name, arguments) -> str:
    return dispatch(name, arguments)
