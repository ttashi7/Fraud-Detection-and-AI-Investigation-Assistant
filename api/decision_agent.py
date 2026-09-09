def make_decision(score):

    if score >= 0.75:
        return "BLOCK"

    elif score >= 0.40:
        return "INVESTIGATE"

    else:
        return "APPROVE"
