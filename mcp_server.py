from __future__ import annotations
from mcp.server.fastmcp import FastMCP  # <--- FIX: Correct import for 'pip install mcp'

# Initialize the server
mcp = FastMCP("arithmetic")


# Helper to ensure numbers are floats
def as_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            pass
    raise TypeError(f"Cannot convert {x} to number")


# --- Define Tools ---


@mcp.tool()
async def add(a: float, b: float) -> float:
    """Add two numbers (a + b)"""
    return as_number(a) + as_number(b)


@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Subtract b from a (a - b)"""
    return as_number(a) - as_number(b)


@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers (a * b)"""
    return as_number(a) * as_number(b)


@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Divide a by b (a / b)"""
    if as_number(b) == 0:
        return "Error: Division by zero"
    return as_number(a) / as_number(b)


@mcp.tool()
async def power(a: float, b: float) -> float:
    """Calculate power (a^b)"""
    return as_number(a) ** as_number(b)


@mcp.tool()
async def modulus(a: float, b: float) -> float:
    """Calculate modulus (a % b)"""
    return as_number(a) % as_number(b)


# --- CRITICAL FIX: Run the Server ---
if __name__ == "__main__":
    mcp.run()
