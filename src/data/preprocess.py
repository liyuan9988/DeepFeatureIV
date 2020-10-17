
def rescale_treatment(treatment, data_name: str):
    if data_name in ["demand", "demand_image"]:
        psd = 3.7
        pmu = 17.779
        return (treatment - pmu) / psd
    else:
        return treatment


def rescale_outcome(outcome, data_name: str):
    if data_name in ["demand", "demand_image"]:
        ysd = 158
        ymu = -292.1
        return (outcome - ymu) / ysd
    else:
        return outcome


def inv_rescale_outcome(predict, data_name: str):
    if data_name in ["demand", "demand_image"]:
        ysd = 158
        ymu = -292.1
        return (predict * ysd) + ymu
    else:
        return predict
