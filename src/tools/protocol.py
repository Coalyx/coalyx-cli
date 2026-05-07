from src.tools.base import tool
from src.tools.registry import register_tool

@tool(name="lsp", description="Query Language Server Protocol (LSP) for definition/references.")
def lsp(action: str, filepath: str, line: int = 0) -> str:
    return f"Stub: LSP {action} executed on {filepath}:{line}. (Feature in development)"

@tool(name="mcp", description="Interact with Model Context Protocol (MCP) servers.")
def mcp(server: str, method: str, params: dict) -> str:
    return f"Stub: Sent {method} to MCP server {server}. (Feature in development)"

@tool(name="list_mcp_resources", description="List available resources from an MCP server.")
def list_mcp_resources(server: str) -> str:
    return f"Stub: Listing resources for MCP server {server}."

@tool(name="read_mcp_resource", description="Read a specific resource from an MCP server.")
def read_mcp_resource(server: str, resource_uri: str) -> str:
    return f"Stub: Read resource {resource_uri} from MCP server {server}."

for t in [lsp, mcp, list_mcp_resources, read_mcp_resource]:
    register_tool(t)
