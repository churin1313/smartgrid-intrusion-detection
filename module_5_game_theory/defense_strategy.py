def select_strategy(risk_score):

    if risk_score < 0.3:
        return "Monitor"

    elif risk_score < 0.7:
        return "Trigger Alert"

    elif risk_score < 0.9:
        return "Isolate Node"

    else:
        return "Reconfigure Grid"
