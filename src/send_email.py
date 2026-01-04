import smtplib
from email.message import EmailMessage
import datetime 
import os 

def send_email(text: str) -> None:
    # Create the email
    msg = EmailMessage()
    msg.set_content(text)
    msg["Subject"] = "Optimal Allocation for {}".format(datetime.datetime.now().strftime("%Y-%m-%d"))
    msg["From"] = os.environ["GMAIL_USER"]
    msg["To"] = "sergio.angelini@postgrad.manchester.ac.uk"

    # Send via Gmail SMTP
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(os.environ["GMAIL_USER"], os.environ["GMAIL_PORTFOLIO_OPT_PASSWORD"])
        server.send_message(msg)

    print("Email sent")