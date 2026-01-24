"""Balancer stable pool math.

Core math functions for stable (StableSwap/Curve-style) pools.
Uses Newton-Raphson iteration for invariant calculation.

IMPORTANT: All financial calculations use SafeInt for overflow protection
and explicit bounds checking.
"""

from solver.math.fixed_point import AMP_PRECISION, Bfp
from solver.models.types import normalize_address
from solver.safe_int import S

from .errors import StableGetBalanceDidNotConverge, StableInvariantDidNotConverge, ZeroBalanceError
from .pools import BalancerStablePool

# Maximum iterations for Newton-Raphson convergence
_STABLE_MAX_ITERATIONS = 255


def calculate_invariant(amp: int, balances: list[Bfp]) -> Bfp:
    """Calculate StableSwap invariant D using Newton-Raphson iteration.

    Uses Balancer's parameterization where the Newton-Raphson formula uses
    A*n (not A*n^n). The n^n factor is incorporated through the iterative
    d_p calculation.

    Algorithm:
        1. Initial guess: D = sum(balances)
        2. Iterate using Newton-Raphson until |D_new - D_old| <= 1 wei
        3. Max iterations: 255

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of token balances (already scaled to 18 decimals)

    Returns:
        The calculated invariant D as Bfp

    Raises:
        StableInvariantDidNotConverge: If iteration doesn't converge
        ZeroBalanceError: If any balance is zero
    """
    n_coins = len(balances)
    if n_coins == 0:
        return Bfp(0)

    # Check for zero balances
    for i, bal in enumerate(balances):
        if bal.value <= 0:
            raise ZeroBalanceError(f"Balance at index {i} must be positive")

    # Calculate sum of balances using SafeInt for overflow protection
    sum_balances = S(sum(b.value for b in balances))

    # Initial guess: D = sum(balances)
    d_prev = sum_balances

    # amp_times_n = A * n (Balancer convention, NOT A * n^n)
    # Note: amp already includes AMP_PRECISION factor
    amp_times_n = S(amp) * S(n_coins)

    for _ in range(_STABLE_MAX_ITERATIONS):
        # d_p = D^(n+1) / (n^n * prod(balances))
        # Computed iteratively: d_p = D, then d_p = d_p * D / (n * balance_i) for each i
        d_p = d_prev
        for bal in balances:
            # d_p = d_p * D / (n * balance)
            # Using integer division that rounds down
            d_p = (d_p * d_prev) // (S(n_coins) * S(bal.value))

        # Newton-Raphson numerator:
        # From Rust: ((ann * sum / AMP_PRECISION + d_p * n_coins) * d)
        term1 = (amp_times_n * sum_balances) // S(AMP_PRECISION)
        numerator = (term1 + d_p * S(n_coins)) * d_prev

        # Newton-Raphson denominator:
        # From Rust: ((ann - AMP_PRECISION) * d / AMP_PRECISION + (n_coins + 1) * d_p)
        term2 = ((amp_times_n - S(AMP_PRECISION)) * d_prev) // S(AMP_PRECISION)
        denominator = term2 + S(n_coins + 1) * d_p

        # d_new = numerator / denominator
        d_new = numerator // denominator

        # Check convergence: |d_new - d_prev| <= 1
        if d_new > d_prev:
            if d_new - d_prev <= 1:
                return Bfp(d_new.value)
        else:
            if d_prev - d_new <= 1:
                return Bfp(d_new.value)

        d_prev = d_new

    raise StableInvariantDidNotConverge(
        f"Stable invariant did not converge after {_STABLE_MAX_ITERATIONS} iterations"
    )


def get_token_balance_given_invariant_and_all_other_balances(
    amp: int,
    balances: list[Bfp],
    invariant: Bfp,
    token_index: int,
) -> Bfp:
    """Solve for balance[token_index] given D and all other balances.

    Uses Newton-Raphson iteration to find y (the unknown balance) such that
    the StableSwap invariant is preserved.

    This implementation exactly matches Balancer's StableMath.sol:
    https://github.com/balancer-labs/balancer-v2-monorepo/blob/stable-deployment/pkg/pool-stable/contracts/StableMath.sol#L465-L516

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of token balances (the value at token_index will be used in c calculation)
        invariant: The invariant D to preserve
        token_index: Index of the token whose balance we're solving for

    Returns:
        The calculated balance as Bfp

    Raises:
        StableGetBalanceDidNotConverge: If iteration doesn't converge
        IndexError: If token_index is out of range
    """
    n_coins = len(balances)
    if token_index < 0 or token_index >= n_coins:
        raise IndexError(f"token_index {token_index} out of range for {n_coins} tokens")

    d = S(invariant.value)
    amp_times_total = S(amp) * S(n_coins)

    # Calculate P_D and sum using SafeInt for overflow protection
    # P_D starts as balance[0] * n
    # For each subsequent balance j: P_D = P_D * balance[j] * n / invariant
    sum_balances = S(balances[0].value)
    p_d = S(balances[0].value) * S(n_coins)

    for j in range(1, n_coins):
        # P_D = Math.divDown(Math.mul(Math.mul(P_D, balances[j]), balances.length), invariant)
        p_d = (p_d * S(balances[j].value) * S(n_coins)) // d
        sum_balances = sum_balances + S(balances[j].value)

    # sum = sum - balances[token_index]
    sum_others = sum_balances - S(balances[token_index].value)

    inv2 = d * d

    # c = inv2 / (ampTimesTotal * P_D) * AMP_PRECISION * balances[tokenIndex]
    # Using div_up for the first division
    amp_times_p_d = amp_times_total * p_d
    # div_up: (inv2 + amp_times_p_d - 1) // amp_times_p_d
    if amp_times_p_d == 0:
        raise StableGetBalanceDidNotConverge("amp_times_p_d is zero")
    c_step1 = (inv2 + amp_times_p_d - S(1)) // amp_times_p_d  # div_up
    c = c_step1 * S(AMP_PRECISION) * S(balances[token_index].value)

    # b = sum_others + invariant / ampTimesTotal * AMP_PRECISION
    # Using div_down for the division
    b = sum_others + (d // amp_times_total) * S(AMP_PRECISION)

    # Initial guess: tokenBalance = (inv2 + c) / (invariant + b)
    # Using div_up
    numerator_init = inv2 + c
    denominator_init = d + b
    token_balance = (numerator_init + denominator_init - S(1)) // denominator_init  # div_up

    for _ in range(_STABLE_MAX_ITERATIONS):
        prev_token_balance = token_balance

        # tokenBalance = (tokenBalance² + c) / (2*tokenBalance + b - invariant)
        # Using div_up
        numerator = token_balance * token_balance + c
        denominator = S(2) * token_balance + b - d

        if denominator <= 0:
            raise StableGetBalanceDidNotConverge("Denominator became non-positive")

        token_balance = (numerator + denominator - S(1)) // denominator  # div_up

        # Check convergence: |token_balance - prev_token_balance| <= 1
        if token_balance > prev_token_balance:
            if token_balance - prev_token_balance <= 1:
                return Bfp(token_balance.value)
        else:
            if prev_token_balance - token_balance <= 1:
                return Bfp(token_balance.value)

    raise StableGetBalanceDidNotConverge(
        f"Stable get_balance did not converge after {_STABLE_MAX_ITERATIONS} iterations"
    )


def stable_calc_out_given_in(
    amp: int,
    balances: list[Bfp],
    token_index_in: int,
    token_index_out: int,
    amount_in: Bfp,
) -> Bfp:
    """Calculate output amount for a given input in a stable pool.

    Fee should be subtracted from amount_in BEFORE calling this function.
    Unlike weighted pools, stable pools do not enforce ratio limits.

    Algorithm:
        1. Calculate current invariant D
        2. Add amount_in to balances[token_index_in]
        3. Solve for new balances[token_index_out] given D
        4. Return: old_balance_out - new_balance_out - 1 (1 wei rounding protection)

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of scaled token balances (18 decimals)
        token_index_in: Index of input token
        token_index_out: Index of output token
        amount_in: Scaled input amount (after fee subtraction)

    Returns:
        Scaled output amount

    Raises:
        StableInvariantDidNotConverge: If invariant calculation doesn't converge
        StableGetBalanceDidNotConverge: If balance calculation doesn't converge
        ValueError: If token_index_in == token_index_out
        IndexError: If token indices are out of range
    """
    # Validate indices
    n_coins = len(balances)
    if token_index_in < 0 or token_index_in >= n_coins:
        raise IndexError(f"token_index_in {token_index_in} out of range for {n_coins} tokens")
    if token_index_out < 0 or token_index_out >= n_coins:
        raise IndexError(f"token_index_out {token_index_out} out of range for {n_coins} tokens")
    if token_index_in == token_index_out:
        raise ValueError("Cannot swap token with itself")

    # Calculate current invariant
    invariant = calculate_invariant(amp, balances)

    # Create new balances list with updated input balance
    new_balances = list(balances)
    new_balances[token_index_in] = Bfp(balances[token_index_in].value + amount_in.value)

    # Calculate new output balance that preserves invariant
    new_balance_out = get_token_balance_given_invariant_and_all_other_balances(
        amp, new_balances, invariant, token_index_out
    )

    # Output = old_balance_out - new_balance_out - 1 (rounding protection)
    old_balance_out = balances[token_index_out].value
    new_balance_out_val = new_balance_out.value

    # Ensure we don't underflow
    if new_balance_out_val >= old_balance_out:
        return Bfp(0)

    amount_out = old_balance_out - new_balance_out_val - 1
    return Bfp(amount_out)


def stable_calc_in_given_out(
    amp: int,
    balances: list[Bfp],
    token_index_in: int,
    token_index_out: int,
    amount_out: Bfp,
) -> Bfp:
    """Calculate input amount for a given output in a stable pool.

    Fee should be added to the result AFTER calling this function.
    Unlike weighted pools, stable pools do not enforce ratio limits.

    Algorithm:
        1. Calculate current invariant D
        2. Subtract amount_out from balances[token_index_out]
        3. Solve for new balances[token_index_in] given D
        4. Return: new_balance_in - old_balance_in + 1 (1 wei rounding protection)

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of scaled token balances (18 decimals)
        token_index_in: Index of input token
        token_index_out: Index of output token
        amount_out: Scaled output amount

    Returns:
        Scaled input amount (before fee addition)

    Raises:
        StableInvariantDidNotConverge: If invariant calculation doesn't converge
        StableGetBalanceDidNotConverge: If balance calculation doesn't converge
        ZeroBalanceError: If amount_out >= balance_out
        ValueError: If token_index_in == token_index_out
        IndexError: If token indices are out of range
    """
    # Validate indices
    n_coins = len(balances)
    if token_index_in < 0 or token_index_in >= n_coins:
        raise IndexError(f"token_index_in {token_index_in} out of range for {n_coins} tokens")
    if token_index_out < 0 or token_index_out >= n_coins:
        raise IndexError(f"token_index_out {token_index_out} out of range for {n_coins} tokens")
    if token_index_in == token_index_out:
        raise ValueError("Cannot swap token with itself")

    # Check that we're not requesting more than available
    if amount_out.value >= balances[token_index_out].value:
        raise ZeroBalanceError("amount_out must be less than balance_out")

    # Calculate current invariant
    invariant = calculate_invariant(amp, balances)

    # Create new balances list with updated output balance
    new_balances = list(balances)
    new_balances[token_index_out] = Bfp(balances[token_index_out].value - amount_out.value)

    # Calculate new input balance that preserves invariant
    new_balance_in = get_token_balance_given_invariant_and_all_other_balances(
        amp, new_balances, invariant, token_index_in
    )

    # Input = new_balance_in - old_balance_in + 1 (rounding protection)
    old_balance_in = balances[token_index_in].value
    new_balance_in_val = new_balance_in.value

    amount_in = new_balance_in_val - old_balance_in + 1
    return Bfp(amount_in)


def filter_bpt_token(pool: BalancerStablePool) -> BalancerStablePool:
    """Filter out BPT token from composable stable pool reserves.

    For composable stable pools, the pool's own BPT token is included
    in the reserves but must be filtered out before calculations.

    Detection: BPT token address == pool address

    Args:
        pool: The stable pool to filter

    Returns:
        A new pool with BPT token removed from reserves (if present)
    """
    pool_address_norm = normalize_address(pool.address)
    filtered_reserves = tuple(
        r for r in pool.reserves if normalize_address(r.token) != pool_address_norm
    )

    # If nothing was filtered, return original pool
    if len(filtered_reserves) == len(pool.reserves):
        return pool

    # Return new pool with filtered reserves
    return BalancerStablePool(
        id=pool.id,
        address=pool.address,
        pool_id=pool.pool_id,
        reserves=filtered_reserves,
        amplification_parameter=pool.amplification_parameter,
        fee=pool.fee,
        gas_estimate=pool.gas_estimate,
    )
