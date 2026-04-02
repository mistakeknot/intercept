# lib-intercept.sh — sourceable wrapper for intercept gates in shell hooks
# Usage: source this file, then call intercept_decide
#
#   source lib-intercept.sh
#   decision=$(intercept_decide convergence-gate "$input_json")
#   if [[ "$decision" == "PROCEED" ]]; then ...

[[ -n "${_LIB_INTERCEPT_LOADED:-}" ]] && return 0
_LIB_INTERCEPT_LOADED=1

_INTERCEPT_BIN="${_INTERCEPT_BIN:-$(dirname "$(dirname "${BASH_SOURCE[0]}")")/bin/intercept}"

# Make a gate decision.
# Args: $1=gate-name, $2=input-json
# Returns: decision string on stdout (e.g., PROCEED, SKIP)
intercept_decide() {
    local gate="${1:?gate name required}"
    local input="${2:?input json required}"

    if [[ -x "$_INTERCEPT_BIN" ]]; then
        "$_INTERCEPT_BIN" decide "$gate" --input "$input" 2>/dev/null
    else
        # intercept not installed — return gate default
        echo "PROCEED"
    fi
}

# Check gate status.
# Args: $1=gate-name (optional)
intercept_status() {
    local gate="${1:-}"
    if [[ -x "$_INTERCEPT_BIN" ]]; then
        "$_INTERCEPT_BIN" status $gate 2>/dev/null
    else
        echo "intercept not installed"
    fi
}
