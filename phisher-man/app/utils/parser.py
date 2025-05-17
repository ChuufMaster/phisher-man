from email import message_from_string
from email.header import decode_header


def decode_header_value(value):
    if value is None:
        return ""
    decoded_parts = decode_header(value)
    return "".join(
        part.decode(encoding or "utf-8") if isinstance(part, bytes) else part
        for part, encoding in decoded_parts
    )


def extract_body(message):
    if not message.is_multipart():
        return message.get_payload(decode=True).decode(errors="replace")

    for part in message.walk():
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition"))

        if content_type == "text/plain" and "attachment" not in content_disposition:
            return part.get_payload(decode=True).decode(errors="replace")

    return ""


def parse_email_raw(raw_email: str):
    msg = message_from_string(raw_email)
    sender = decode_header_value(msg.get("From"))
    receiver = decode_header_value(msg.get("To"))
    subject = decode_header_value(msg.get("subject"))
    body = extract_body(msg)

    return {"sender": sender, "receiver": receiver, "subject": subject, "body": body}
