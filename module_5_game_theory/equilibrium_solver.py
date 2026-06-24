from utility_function import payoff_matrix


def find_best_defense():

    best_action = None
    best_payoff = float("-inf")

    for (_, defense), (_, defender_payoff) in payoff_matrix.items():

        if defender_payoff > best_payoff:
            best_payoff = defender_payoff
            best_action = defense

    return best_action
