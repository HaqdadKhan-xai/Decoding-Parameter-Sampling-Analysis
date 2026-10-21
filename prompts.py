from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptDefinition:
    """A stable experiment prompt and its reference metadata."""

    prompt_id: str
    objective: str
    text: str
    criteria: tuple[str, ...] = ()
    expected_output: dict[str, Any] | None = None


PROMPTS = (
    PromptDefinition(
        "D1",
        "variability",
        "Explain how historical electricity use, weather, and calendar effects help "
        "forecast demand. State two benefits and one limitation.",
        (
            "historical consumption",
            "weather",
            "calendar effects",
            "two benefits",
            "one limitation",
        ),
    ),
    PromptDefinition(
        "F1",
        "factual",
        "Explain MAE, RMSE, and MAPE in forecasting. State how each is calculated, "
        "how RMSE differs from MAE, and one limitation of MAPE.",
        (
            "MAE is mean absolute error based on absolute actual-predicted differences.",
            "RMSE is the square root of mean squared errors.",
            "RMSE gives larger errors greater influence than MAE.",
            "MAPE is mean absolute percentage error, usually expressed as a percentage.",
            "MAPE is undefined or unstable when actual values are zero or near zero.",
            "The distinctions between MAE, RMSE, and MAPE are correct.",
        ),
    ),
    PromptDefinition(
        "S1",
        "structure",
        "Return only valid JSON with keys actual, predicted, and error. Actual demand "
        "is 500 MW and predicted demand is 450 MW. Set error to the absolute "
        "difference. Use numbers, not strings.",
        expected_output={"actual": 500, "predicted": 450, "error": 50},
    ),
)

PROMPTS_BY_ID = {prompt.prompt_id: prompt for prompt in PROMPTS}


def validate_prompts() -> None:
    """Raise ValueError if the revised prompt set is malformed."""
    expected_ids = {"D1", "F1", "S1"}
    if len(PROMPTS) != 3 or set(PROMPTS_BY_ID) != expected_ids:
        raise ValueError("The experiment must contain exactly D1, F1, and S1.")
    if any(not prompt.text.strip() for prompt in PROMPTS):
        raise ValueError("Every prompt must contain non-empty text.")
