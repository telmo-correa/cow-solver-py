"""SwapRouterV2 calldata encoding for UniswapV3 swaps."""

from __future__ import annotations

from solver.models.types import normalize_address

from .constants import SWAP_ROUTER_V2_ADDRESS

# Function selectors for SwapRouterV2
# exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))
EXACT_INPUT_SINGLE_SELECTOR = bytes.fromhex("04e45aaf")

# exactOutputSingle((address,address,uint24,address,uint256,uint256,uint160))
EXACT_OUTPUT_SINGLE_SELECTOR = bytes.fromhex("5023b4df")

# SwapRouterV2 ABI - minimal, just the functions we need
SWAP_ROUTER_V2_ABI = [
    {
        "name": "exactInputSingle",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "recipient", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMinimum", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [{"name": "amountOut", "type": "uint256"}],
    },
    {
        "name": "exactOutputSingle",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "recipient", "type": "address"},
                    {"name": "amountOut", "type": "uint256"},
                    {"name": "amountInMaximum", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [{"name": "amountIn", "type": "uint256"}],
    },
]


def encode_exact_input_single(
    token_in: str,
    token_out: str,
    fee: int,
    recipient: str,
    amount_in: int,
    amount_out_minimum: int,
    sqrt_price_limit_x96: int = 0,
) -> tuple[str, str]:
    """Encode SwapRouterV2.exactInputSingle call.

    Args:
        token_in: Input token address
        token_out: Output token address
        fee: Pool fee tier (e.g., 3000 for 0.3%)
        recipient: Address to receive output tokens
        amount_in: Amount of input tokens
        amount_out_minimum: Minimum output amount (slippage protection)
        sqrt_price_limit_x96: Price limit (0 = no limit)

    Returns:
        Tuple of (router_address, calldata_hex)
    """
    from eth_abi import encode  # type: ignore[attr-defined]

    # Normalize addresses
    token_in_normalized = normalize_address(token_in)
    token_out_normalized = normalize_address(token_out)
    recipient_normalized = normalize_address(recipient)

    # Convert to bytes for encoding
    token_in_bytes = bytes.fromhex(token_in_normalized[2:])
    token_out_bytes = bytes.fromhex(token_out_normalized[2:])
    recipient_bytes = bytes.fromhex(recipient_normalized[2:])

    # Encode the struct as a tuple
    # (address tokenIn, address tokenOut, uint24 fee, address recipient,
    #  uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96)
    encoded_params = encode(
        ["(address,address,uint24,address,uint256,uint256,uint160)"],
        [
            (
                token_in_bytes,
                token_out_bytes,
                fee,
                recipient_bytes,
                amount_in,
                amount_out_minimum,
                sqrt_price_limit_x96,
            )
        ],
    )

    calldata = EXACT_INPUT_SINGLE_SELECTOR + encoded_params
    return SWAP_ROUTER_V2_ADDRESS, "0x" + calldata.hex()


def encode_exact_output_single(
    token_in: str,
    token_out: str,
    fee: int,
    recipient: str,
    amount_out: int,
    amount_in_maximum: int,
    sqrt_price_limit_x96: int = 0,
) -> tuple[str, str]:
    """Encode SwapRouterV2.exactOutputSingle call.

    Args:
        token_in: Input token address
        token_out: Output token address
        fee: Pool fee tier (e.g., 3000 for 0.3%)
        recipient: Address to receive output tokens
        amount_out: Exact amount of output tokens desired
        amount_in_maximum: Maximum input amount (slippage protection)
        sqrt_price_limit_x96: Price limit (0 = no limit)

    Returns:
        Tuple of (router_address, calldata_hex)
    """
    from eth_abi import encode  # type: ignore[attr-defined]

    # Normalize addresses
    token_in_normalized = normalize_address(token_in)
    token_out_normalized = normalize_address(token_out)
    recipient_normalized = normalize_address(recipient)

    # Convert to bytes for encoding
    token_in_bytes = bytes.fromhex(token_in_normalized[2:])
    token_out_bytes = bytes.fromhex(token_out_normalized[2:])
    recipient_bytes = bytes.fromhex(recipient_normalized[2:])

    # Encode the struct as a tuple
    # (address tokenIn, address tokenOut, uint24 fee, address recipient,
    #  uint256 amountOut, uint256 amountInMaximum, uint160 sqrtPriceLimitX96)
    encoded_params = encode(
        ["(address,address,uint24,address,uint256,uint256,uint160)"],
        [
            (
                token_in_bytes,
                token_out_bytes,
                fee,
                recipient_bytes,
                amount_out,
                amount_in_maximum,
                sqrt_price_limit_x96,
            )
        ],
    )

    calldata = EXACT_OUTPUT_SINGLE_SELECTOR + encoded_params
    return SWAP_ROUTER_V2_ADDRESS, "0x" + calldata.hex()


__all__ = [
    "EXACT_INPUT_SINGLE_SELECTOR",
    "EXACT_OUTPUT_SINGLE_SELECTOR",
    "SWAP_ROUTER_V2_ABI",
    "encode_exact_input_single",
    "encode_exact_output_single",
]
