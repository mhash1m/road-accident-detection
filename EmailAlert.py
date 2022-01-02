import smtplib
from email.message import EmailMessage

def send_Alert(to):
    msg = EmailMessage()
    msg.set_content("An accident has been detected")
    msg['subject'] = "Accident detection alert"
    msg['to'] = to


    user = "accident.detection.alert@gmail.com"
    password = "vseungjvvhtzmbfe"

    msg['from'] = user
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    server.quit()