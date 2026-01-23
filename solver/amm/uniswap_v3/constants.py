"""UniswapV3 constants including fee tiers and contract addresses."""

# V3 Fee tiers in Uniswap units (hundredths of a basis point)
# Fee = units / 1,000,000 (e.g., 3000 = 0.3%)
V3_FEE_LOWEST = 100  # 0.01% - stable pairs
V3_FEE_LOW = 500  # 0.05% - stable pairs
V3_FEE_MEDIUM = 3000  # 0.30% - most pairs
V3_FEE_HIGH = 10000  # 1.00% - exotic pairs

V3_FEE_TIERS = [V3_FEE_LOWEST, V3_FEE_LOW, V3_FEE_MEDIUM, V3_FEE_HIGH]

# Tick spacing per fee tier
V3_TICK_SPACING = {
    V3_FEE_LOWEST: 1,
    V3_FEE_LOW: 10,
    V3_FEE_MEDIUM: 60,
    V3_FEE_HIGH: 200,
}

# Gas cost for V3 swaps (from Rust solver)
V3_SWAP_GAS_COST = 106_000

# Contract addresses (mainnet)
QUOTER_V2_ADDRESS = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
SWAP_ROUTER_V2_ADDRESS = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"

__all__ = [
    "V3_FEE_LOWEST",
    "V3_FEE_LOW",
    "V3_FEE_MEDIUM",
    "V3_FEE_HIGH",
    "V3_FEE_TIERS",
    "V3_TICK_SPACING",
    "V3_SWAP_GAS_COST",
    "QUOTER_V2_ADDRESS",
    "SWAP_ROUTER_V2_ADDRESS",
]
