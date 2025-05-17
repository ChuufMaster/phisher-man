from pydantic import BaseModel, constr, EmailStr


class EmailInput(BaseModel):
    """
    This represents the allowed input for the predict endoint and sets the
    limits on the sizes of inputs

    Attributes:
        sender (str): Address of email sender, limited to max length of an email address
        receiver (str): Address of email receiver, limited to max length of an email address
        subject (str): According the the RFC2822 a line should at most be 998 characters long
        body: The max length should be based on one standard deviation of character count for an email body
    """

    sender: constr(strip_whitespace=True, min_length=1, max_length=256)
    receiver: constr(strip_whitespace=True, min_length=1, max_length=256)
    subject: constr(strip_whitespace=True, min_length=1, max_length=998)
    body: constr(strip_whitespace=True, min_length=1, max_length=20000)


class PredictionOutput(BaseModel):
    """
    This represents the output from the prediciton

    Attributes:
        prediction (str): Either PHISHING or LEGITIMATE
        confidence (str): The level of confidence of phishing, higher means more likely phishing
        reasons [str, str, float]: The top 5 contributing factors and the amount it effected the result
    """

    prediction: str
    confidence: str
    reasons: str
